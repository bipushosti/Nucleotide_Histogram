#include <stdio.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <limits.h>
#include <string>

#include <vector>
#include <ctype.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define THREADS_PER_BLOCK	32
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;

static void HandleError( cudaError_t err,const char *file, int line);

__global__ void create_number(char* sequence,
	 		uint32_t* number_values,
			uint32_t max_gene_length,
			uint32_t search_sequence_length,
			uint32_t total_numbers);

uint32_t* get_threadsPerBlock_numberConversion (uint32_t max_gene_length,
						uint32_t search_sequence_length,
						uint32_t total_numbers,
						uint32_t total_gene_sequences);

__global__ void get_count (uint32_t* gene_sequence_numbers,
	 		uint32_t* search_number_values,
			uint16_t* count,
			uint32_t total_gene_sequences,
			uint32_t total_search_sequences,
			uint32_t numbers_per_gene);

//************************************************************************************************

__global__ void get_count (uint32_t* gene_sequence_numbers,
	 		uint32_t* search_number_values,
			uint16_t* count,
			uint32_t total_gene_sequences,
			uint32_t total_search_sequences,
			uint32_t numbers_per_gene)
{

	uint32_t blockId = blockIdx.y * gridDim.x + blockIdx.x;	
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t id = blockId * blockDim.x + threadIdx.x;

	uint32_t i=0;
	uint16_t count_value;

	if(idx < total_search_sequences) {

		count_value = 0;

		for(i=0; i<numbers_per_gene; i++) {
			//printf("Search Number is: %d\n", search_number_values[idx]);
			if(search_number_values[idx] == gene_sequence_numbers[idy * numbers_per_gene + i]) {
				count_value++;
			}
		}

		count[idx * total_gene_sequences + idy] = count_value;
/*
		if((idy == 0) || (idy == 1)) {
			printf("Total Search Sequences: %d Idx: %d Idy: %d Sequence: %d Count: %d\n", total_search_sequences,idx, idy, search_number_values[idx], count[idx*total_gene_sequences + idy]);
		}*/
	}

}


//----------------------------------------------------------------------------------------------//
__global__ void create_number(char* gene_sequence,
	 		uint32_t* number_values,
			uint32_t max_gene_length,
			uint32_t search_sequence_length,
			uint32_t total_numbers)
{

	uint32_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
	uint32_t id = blockId * blockDim.x + threadIdx.x;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;


	//if(idx < (max_gene_length - search_sequence_length + 1)) {
	if(idx < total_numbers) {

		//char* string;
		uint32_t number = 0;


		uint16_t i,j;
		uint8_t k;
		j=id;
		k=0;


		uint32_t multiplier = 1;


		//pow() function does not give exact result
		//This is basically getting pow(10, search_sequence_length)
		for(i=1; i< search_sequence_length; i++) {
			multiplier *= 10;
		}


		for(i=0; i < search_sequence_length; i++, j++) {

			//printf("Number at start is: %d\n",number);

		//for(i=idx; i<(idx+search_sequence_length); i++,j++,k++)

			switch(gene_sequence[j]) {
				case ('A'):
					//string[i] = '1';
					number+= 1 * multiplier;
					break;
				case ('T'):
					//string[i] = '2';
					number+= 2 * multiplier;
					break;
				case ('G'):
					//string[i] = '3';
					number+= 3 * multiplier;
					break;
				case ('C'):
					//string[i] = '4';
					number+= 4 * multiplier;
					break;
				case ('a'):
					//string[i] = '1';
					number+= 1 * multiplier;
					break;
				case ('t'):
					//string[i] = '1';
					number+= 2 * multiplier;
					break;
				case ('g'):
					//string[i] = '1';
					number+= 3 * multiplier;
					break;
				case ('c'):
					//string[i] = '1';
					number+= 4 * multiplier;
					break;
				case ('N'):
					//string[i] = '7';
					number+= 7 * multiplier;
					break;
				case ('n'):
					//string[i] = '7';
					number+= 7 * multiplier;
					break;
				case ('\0'):
					//string[i] = '9';
					number+= 9 * multiplier;
					break;
				default:
					number+= 9 * multiplier;
			}

			multiplier = multiplier / 10;

			//if(idx == 0) {
				//printf("Number is: %d \t Multiplier value is: %d\n", number, multiplier );
			//}
		}
		//scanf(string,"%d",&number);
		//number = atoi(string);


		number_values[id] = number;
		//printf("Total Numbers is: %d BlockDim.x: %d Thread Y ID: %d Thread X Id: %d Number is: %d\n",total_numbers, blockDim.x, idy,idx,number);
	}
}


template <typename Vector1, typename Vector2>
void dense_histogram(const Vector1& input, Vector2& histogram)
{

	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector2::value_type IndexType; // histogram index type

	// copy input data (could be skipped if input is allowed to be modified)
	thrust::device_vector<ValueType> data(input);

	  // print the initial data
  	print_vector("initial data", data);


	// sort data to bring equal elements together
	thrust::sort(data.begin(), data.end());

	// number of histogram bins is equal to the maximum value plus one
	IndexType num_bins = data.back() + 1;

	// resize histogram storage
	histogram.resize(num_bins);

	// find the end of each bin of values
	thrust::counting_iterator<IndexType> search_begin(0);
	thrust::upper_bound(data.begin(), data.end(),
		      search_begin, search_begin + num_bins,
		      histogram.begin());

	// print the cumulative histogram
	print_vector("cumulative histogram", histogram);

	// compute the histogram by taking differences of the cumulative histogram
	thrust::adjacent_difference(histogram.begin(), histogram.end(),
		              histogram.begin());

	// print the histogram
	print_vector("histogram", histogram);
}



template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
	typedef typename Vector::value_type T;

	std::cout << "  " << std::setw(20) << name << "  ";

	thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));

	std::cout << std::endl;
}


static void HandleError( cudaError_t err,const char *file, int line)
{
	if (err != cudaSuccess)
	{
	        fprintf( stderr,"%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
		exit(err);
    	}
}

//Function that gets the total threads per block required for the number calculation kernel
uint32_t* get_threadsPerBlock_numberConversion (uint32_t max_gene_length,
						uint32_t search_sequence_length,
						uint32_t total_numbers,
						uint32_t total_gene_sequences)
{
	uint32_t threadsPBlock;
	uint32_t blocks_x;
	static uint32_t returnArr[3];

	if(max_gene_length > 1024) {
		threadsPBlock = 1024;
		blocks_x = (max_gene_length + 1023) / 1024;
	}
	else if ((max_gene_length < 1024) && (max_gene_length > 512)) {
		threadsPBlock = 512;
		blocks_x = (max_gene_length + 511) / 512;
	}
	else {
		threadsPBlock = 32;
		blocks_x = (max_gene_length + 31) / 32;
	}

	returnArr[0] = threadsPBlock;
	returnArr[1] = blocks_x;
	returnArr[2] = total_gene_sequences;

	return returnArr;

}

int main(int argc, char* argv[])
{

	/**** User supplied variables *****/

	//Total number of gene sequences in the file
	uint32_t total_gene_sequences;

	//Maximum length of each gene sequence
	uint32_t max_gene_length;

	uint32_t total_search_sequences;

	uint32_t search_sequence_length;





	/**** Host TypeData Arrays  *********/

	//vector<vector<string> > gene_sequences;
	//vector<string> search_sequences;

	//Search sequences in 4 byte integers
	uint32_t * search_sequence_numbers;
	char* gene_sequences;
	uint16_t* count;


	/**** Device TypeData Arrays  *********/

	char* d_gene_sequences;
	uint32_t* d_search_sequence_numbers;
	uint32_t* d_gene_number_sequences;
	uint16_t* d_count;

	/***************  Temporary Variables **********************/

	//Temporary variables needed for loops
	int i,j;

	//Variable needed for getopt; Holds the the option; In "-n 10" holds 'n' and 'optarg' holds 10
	int option;


	/***************  Reading the user arguments ***************/


	//Checking if correct number of arguments were provided
	if(argc != 7){
		printf("Usage: Executable -n Number_of_Genes -m Max_Gene_Length -l Search_Sequence_Length \n");
	 	exit(EXIT_FAILURE);
	}

	//Parsing the options provided
	while((option = getopt(argc,argv,"n:m:l:")) != -1){
		switch (option){

			case 'n':
				total_gene_sequences = atoi(optarg);
				break;
			case 'm':
				max_gene_length = atoi(optarg);
				break;
			case 'l':
				search_sequence_length = atoi(optarg);
				break;
			default:
				printf("Usage: Executable -n Number_of_Genes -m Max_Gene_Length -l Search_Sequence_Length \n");
			 	exit(EXIT_FAILURE);
		}
	}



	//Total number of sequences of size search_sequence_length in each gene
	uint32_t numbers_per_gene = max_gene_length - search_sequence_length + 1;


	/***************  Reading input data from file **********/

/*
	//----------------Reading all genes------------//
	ifstream gene_file("fasta_1.txt");

	string line;

	i=0;

	//Temporary vector to push onto the gene_sequences vector
	vector<string> temp_vector;

	while(getline(gene_file, line)) {

		if(line.at(0) == '>') {
			continue;
		}
		else {
			temp_vector.clear();
			temp_vector.push_back(line);
			gene_sequences.push_back(temp_vector);

		}
		i++;
	}

	cout << "Vector size is: " << gene_sequences.size() << "\n";

*/
	printf("Max Gene Length: %d\n",max_gene_length);

	//Getting the total number of permutations; 4 ^ length of the search sequence
	total_search_sequences = 2 << (search_sequence_length* 2 - 1);



	/***************  Allocating memoery for host arrays *******/

	search_sequence_numbers = (uint32_t*)malloc(total_search_sequences * sizeof(uint32_t));
	gene_sequences = (char*)malloc(total_gene_sequences * max_gene_length * sizeof(char));
	count = (uint16_t*)malloc(total_gene_sequences * total_search_sequences * sizeof(uint32_t)); 

	/***************  Allocating memoery for device arrays *****/

	HANDLE_ERROR(cudaMalloc((void**)&d_search_sequence_numbers, total_search_sequences * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMalloc((void**)&d_gene_sequences, total_gene_sequences * max_gene_length * sizeof(char)));
	HANDLE_ERROR(cudaMalloc((void**)&d_gene_number_sequences, numbers_per_gene * total_gene_sequences * sizeof(uint32_t)));

	/***************  Reading the files for input data *********/

	//Open file where A, T, G, C is replaced by 1, 2, 3 and 4 respectively
	FILE * search_sequences_file;
	FILE * gene_sequences_file;

	search_sequences_file = fopen("all_size_eight_combinations_numbers.txt","r");
	gene_sequences_file = fopen("fasta_test_10_seqs.txt","r");

	char *line;
	char *line2;
	uint32_t integer_value;


	//-----------Reading all combinations----------//
	i=0;

	line = (char*)malloc((search_sequence_length + 10) * sizeof(char));

	while(fgets(line, search_sequence_length + 10, search_sequences_file)) {

		integer_value = atof(line);
		search_sequence_numbers[i] = integer_value;

		i++;
	}

	fclose(search_sequences_file);
	free(line);

	//-----------Reading all genes ----------------//

	//memset(gene_sequences,'0',sizeof(gene_sequences));

	i=0;
	j=0;

	line2 = (char*)malloc((max_gene_length + 2) * sizeof(char));

	memset(line,'\0',max_gene_length + 2);

	while(fgets(line2, max_gene_length + 2, gene_sequences_file)) {


		if(line2[0] == '>') {
			continue;
		}
		//printf("%s\n",line2);
		strncpy(gene_sequences + i,line2,max_gene_length);
		//printf("%s\n",(gene_sequences + i));
/*		for(j=0; j < max_gene_length; j++, i++) {
			gene_sequences[i] = line2[j];
			printf("%c",gene_sequences[j]);
		}
*/
		i+=max_gene_length;
		memset(line,'\0',max_gene_length + 2);
	}


	fclose(gene_sequences_file);


	free(line2);


	/***************  Copying data from Host to Device *****/

	HANDLE_ERROR(cudaMemcpy(d_search_sequence_numbers, search_sequence_numbers, total_search_sequences * sizeof(uint32_t) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_gene_sequences, gene_sequences, total_gene_sequences * max_gene_length * sizeof(char) , cudaMemcpyHostToDevice));



	uint32_t* blockGridDims = get_threadsPerBlock_numberConversion (max_gene_length, search_sequence_length, numbers_per_gene, total_gene_sequences);
	uint32_t threads_per_block = blockGridDims[0];
	uint32_t gridSize_x = blockGridDims[1];
	uint32_t gridSize_y = blockGridDims[2];

	printf("Threads/Block, X GridSize, Y GridSize: %d %d %d\n",threads_per_block, gridSize_x, gridSize_y);


	dim3 blockSize(threads_per_block,1,1);
	dim3 gridSize(gridSize_x, gridSize_y,1);


//	dim3 blockSize(threads_per_block,1,1);
//	dim3 gridSize(3,1,1);


	create_number<<<gridSize, blockSize>>>(d_gene_sequences, d_gene_number_sequences, max_gene_length, search_sequence_length, numbers_per_gene);
	HANDLE_ERROR(cudaDeviceSynchronize());

	//Freeing memory no longer required for calculations
	HANDLE_ERROR(cudaFree(d_gene_sequences));
	free(gene_sequences);

	//---------------------------------------------//
	
	//Allocating memory for the device array that will host the count values
	HANDLE_ERROR(cudaMalloc((void**)&d_count, total_search_sequences * total_gene_sequences * sizeof(uint16_t)));
	HANDLE_ERROR(cudaMemset((void *)d_count, 0, total_search_sequences * total_gene_sequences * sizeof(uint16_t)));
	




	dim3 blockSize2(512,1,1);
	dim3 gridSize2(128,total_gene_sequences,1);

	get_count <<<gridSize2, blockSize2>>>(d_gene_number_sequences, d_search_sequence_numbers, d_count, total_gene_sequences, total_search_sequences,numbers_per_gene);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaMemcpy(count, d_count, total_gene_sequences * total_search_sequences * sizeof(uint16_t), cudaMemcpyDeviceToHost));



	//----------------------------------------------//

	for(i=0;i<total_search_sequences; i++) {

		printf("%d ",search_sequence_numbers[i]);

		for(j=0; j<total_gene_sequences; j++) {
			//count[idx*total_gene_sequences + idy]
			printf("%d ",count[i * total_gene_sequences + j]);
			//printf("%d ",(uint32_t)count[i * total_gene_sequences + j]);
		}
		printf("\n");

	}


	//-----------Reading all combinations----------//
/*
	ifstream combination_file("all_size_eight_combinations.txt");

	string line_search;

	while(getline(combination_file, line_search)) {
		search_sequences.push_back(line_search);
	}

	cout << "Search Vector size is: " << search_sequences.size() << "\n";
	cout<< search_sequences.at(0) << "\n";


	//Creating numbers for each 8 sequence
	uint32_t* number_values;
*/
	//1D Grid with blocksize containing 32 threads
/*	dim3 gridSize(((totalObservations*totalTypes) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1,1);
	dim3 blockSize(32,1,1);

	//calculate_number<<<>>>(gene_string,number_values);
*/
	/***************  Generating Histogram ***************/
/*
	thrust::device_vector<int> histogram;
	thrust:: device_vector <string> vec1;

	vec1 = gene_sequences.at(0);

	print_vector("Test", vec1);
*/
	//dense_histogram(vec1, histogram);


	HANDLE_ERROR(cudaFree(d_search_sequence_numbers));

	HANDLE_ERROR(cudaFree(d_gene_number_sequences));
	HANDLE_ERROR(cudaFree(d_count));

	free(count);
	free(search_sequence_numbers);

}
