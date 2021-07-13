// -> Block matrix multiplication using the Fox's algorithm

void print_matrix(float *A, int N, int M) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M - 1; j++) printf("%.2f, ", A[i * M + j]);
		printf("%.2f\n", A[(i + 1) * M - 1]);
	}
	fflush(stdout);
}

void matrix_multiply(float *local_A, float *local_B, float *local_C, int N, int M) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < M; k++)
				// C(i,j) = A(i,k) * B(k,j)
				local_C[i * N + j] += local_A[i * M + k] * local_B[k * N + j];
}

int main(int argc, char** argv) {
	MPI_Init(NULL, NULL);
	MPI_Comm world = MPI_COMM_WORLD;
	int world_rank;
	MPI_Comm_rank(world, &world_rank);
	int world_size;
	MPI_Comm_size(world, &world_size);

	if ((world_size != 4) && (world_size != 9)) { // We are assuming that there are exactly 4 or 9 processes for this task
		fprintf(stderr, "World size must be equals to 4 or 9 for this task. For example, call like this: mpiexec -n 4 MPI\n");
		MPI_Abort(world, 1);
	}

	MPI_Comm comm;		// Communicator for the entire grid
	MPI_Comm row_comm;	// row-communicator for my row
	MPI_Comm col_comm;	// column-communicator for my column

	int row;			// row coordinate
	int col;			// column coordinate
	int grid_rank;		// rank id in the grid communicator (comm) 

	const int q = (int)sqrt((double)world_size);
	
	int origin[2] = { 0, 0 };			// coordinates origin
	int root_comm;						// rank of the process in the origin of the grid
	int dimensions[2] = { q, q };		// the dimension of the processes' grid is q x q (row x column)
	int coordinates[2];					// to identify each process in the grid
	int wrap_around[2] = { 1, 1 };		// both borders in the grid will be connected (circular) like in a torus, although
	//int wrap_around[2] = { 0, 1 };	// we really only care about a circular shift comunication by columns to share B's local part
	int free_coord[2];					// to get independent communicators so processes can synch by rows or columns independenty

	// ------------------------------------------------------------------------------------------------------------------
	// int MPI_Cart_create(MPI_Comm old_comm, int Ndim, int dims[], int wrap_around, int reorder, MPI_Comm cart_comm) {}
	//
	// old_comm:		base communicator used to build the new communicator
	// Ndim:			number of dimensions, if 2D then Ndim = 2
	// dims:			size of dimensions, e.g. dims[2] = { 2, 3} for a 2 x 3 2D grid of total = 6 processes
	// wrap_around:		to declare whether borders are continuous, e.g. wrap_around[2] = { 1, 1} for the torus case
	// reorder:			set to 1 (true) if wish to reoder processes' rank as possibly the original rank order is not
	//					the most efficient regarding the underlying hardware
	// cart_comm:		cartesian communicator which is created for use among the processes in the grid
	//
	// The processes in cart_comm are ranked in row-major order, e.g. in 2D then the first row consists of processes with
	// ranks 0, 1, ..., dims[1] - 1; then, for the second row, ranks are dims[1], dims[1] + 1, ..., 2 * dims[1] - 1; etc.
	// ------------------------------------------------------------------------------------------------------------------

	MPI_Cart_create(world, 2, dimensions, wrap_around, 1, &comm);
	MPI_Comm_rank(comm, &grid_rank);
	
	// -----------------------------------------------------------------------------
	// int MPI_Cart_rank(MPI_Comm comm, int coordinates[], int rank) {}
	//
	// comm:			communicator
	// coordinates:		coordinates for the current process, e.g. { row, col } in 2D 
	// rank:			return the rank for the process according to its coordinates
	// -----------------------------------------------------------------------------

	MPI_Cart_rank(comm, origin, &root_comm);

	// ----------------------------------------------------------------------------
	// int MPI_Cart_coords(MPI_Comm comm, int rank, int Ndim, int coordinates[]) {}
	//
	// comm:			communicator
	// rank:			process' rank
	// Ndim:			number of dimensions, if 2D then Ndim = 2
	// coordinates:		return the coordinates for the current process, according
	//					to its ranks, e.g. { row, col } in 2D 
	// ----------------------------------------------------------------------------

	MPI_Cart_coords(comm, grid_rank, 2, coordinates);
	row = coordinates[0];
	col = coordinates[1];

	// ----------------------------------------------------------------------------------------
	// int MPI_Cart_sub(MPI_Comm cart_comm, int free_coords[], MPI_Comm *new_comm) {}
	//
	// cart_comm:		cartesian communicator, e.g. in a predefined 2D grid
	// free_coords:		In a 2D grid, if we are creating a communicator shared among
	//					the processes in a row, independently of other rows, then the
	//					1st dimension is fixed (free_coords[0]=0) and the 2nd is
	//					ignored (free_coords[1]=1)
	// new_comm:		return the new communicator
	//
	// MPI_Cart_sub partitions the processes in cart_comm into a collection of disjoint communicators,
	// e.g. grouped by rows or columns in 2D, whose union is cart_comm. Both, cart_comm and new_comm,
	// have associated Cartesian topologies.
	// If cart_comm has dimensions d0 x d1 x ... x dN, then free_coords is an array of N elements.
	// The processes across dimensions that varies (e.g. for which free_coord[i] = 1) share the same
	// splitted communicator in new_comm.
	// ----------------------------------------------------------------------------------------

	// setup row communicator
	free_coord[0] = 0;
	free_coord[1] = 1;
	MPI_Cart_sub(comm, free_coord, &row_comm);

	// setup column communicator
	free_coord[0] = 1;
	free_coord[1] = 0;
	MPI_Cart_sub(comm, free_coord, &col_comm);

	// A and B's dimensions are N x M and M x N, respectively, where N = n x q and M = m x q
	const int n = 2;
	const int m = 3;
	const int N = n * q;
	const int M = m * q;

	// -> Separating memory for local matrix blocks
	float *local_A = (float*)malloc(n * m * sizeof(float));
	float *tmp_A = (float*)malloc(n * m * sizeof(float));
	float *local_B = (float*)malloc(m * n * sizeof(float));
	float *tmp_B = (float*)malloc(m * n * sizeof(float));
	float *local_C = (float*)calloc(n * n, sizeof(float)); // calloc initializes to 0, in contrast to malloc
	float *tmp_C = (float*)malloc(n * n * sizeof(float));

	srand(time(NULL));
	int seed;
	for (int i = 0; i < grid_rank; i++) {
		seed = rand() % (grid_rank * 977 + 9973);
		srand(seed);
	}

	// random initialization of local_A
	for (int i = 0; i < n * m; i++)
		local_A[i] = (float)((int)(10 * (rand() / (RAND_MAX + 1.0))) / 10.0);

	// random initialization of local_B
	for (int i = 0; i < m * n; i++)
		local_B[i] = (float)((int)(10 * (rand() / (RAND_MAX + 1.0))) / 10.0);


	// Every process build their own local parts of A, B and send them to process root_comm for printing input
	if (grid_rank == root_comm) {
		int coords[2];
		int i_rank;
		
		printf("A00 (rank = %d):\n", root_comm);
		print_matrix(local_A, n, m);
		for (int i = 1; i < world_size; i++) {
			coords[0] = i / q; // row index
			coords[1] = i % q; // column index
			MPI_Cart_rank(comm, coords, &i_rank);
			MPI_Recv(tmp_A, n * m, MPI_FLOAT, i_rank, 0, comm, MPI_STATUS_IGNORE);
			printf("\nA%d%d (rank = %d):\n", coords[0], coords[1], i_rank);
			print_matrix(tmp_A, n, m);
		}

		printf("\nB00 (rank = %d):\n", root_comm);
		print_matrix(local_B, m, n);
		for (int i = 1; i < world_size; i++) {
			coords[0] = i / q; // row index
			coords[1] = i % q; // column index
			MPI_Cart_rank(comm, coords, &i_rank);
			MPI_Recv(tmp_B, m * n, MPI_FLOAT, i_rank, 1, comm, MPI_STATUS_IGNORE);
			printf("\nB%d%d (rank = %d):\n", coords[0], coords[1], i_rank);
			print_matrix(tmp_B, m, n);
		}
	}
	else {
		MPI_Send(local_A, n * m, MPI_FLOAT, root_comm, 0, comm);
		MPI_Send(local_B, m * n, MPI_FLOAT, root_comm, 1, comm);
	}

	// -----------------------------------------------------------------------------------------------------------------------
	// int MPI_Sendrecv_replace(void *buffer, int count, MPI_Datatype type, int dest, int send_tag, int source, int recv_tag, 
	//							MPI_Comm comm, MPI_Status *status) {}
	//
	// buffer:		buffer to interchange data
	// count:		number of transferred elements
	// type:		type of transferred elements
	// dest:		rank of process which receives data from current process
	// send_tag:		tag of sent data
	// source:		rank of process which send data to current process
	// recv_tag:		tag of received data
	// comm:		communicator
	// status:		additional information about the transfer
	//
	// The data in buffer is first sent to destination and then received from the source and replaced in buffer. The processes
	// involved in send and receive can be the same. It can be used in conjoint with MPI_Recv and MPI_Send. In contrast with 
	// MPI_Sendrecv, MPI_Sendrecv_replace uses the same buffer for send and receive operations.
	// -----------------------------------------------------------------------------------------------------------------------

	// Fox's algorithm
	int source = (row + 1) % q;
	int dest = (row + q - 1) % q;

	float *tmp = (float*)malloc(m * n * sizeof(float));

	for (int stage = 0; stage < q; stage++) {
		int bcast_root = (row + stage) % q;
		if (bcast_root == col) {
			MPI_Bcast(local_A, n * m, MPI_FLOAT, bcast_root, row_comm);
			matrix_multiply(local_A, local_B, local_C, n, m);
		}
		else {
			MPI_Bcast(tmp_A, n * m, MPI_FLOAT, bcast_root, row_comm);
			matrix_multiply(tmp_A, local_B, local_C, n, m);
		}
		MPI_Sendrecv_replace(local_B, m * n, MPI_FLOAT, dest, 0, source, 0, col_comm, MPI_STATUS_IGNORE);
	}

	// Printing output
	if (grid_rank == root_comm) {
		int coords[2];
		int i_rank;

		printf("\nC00 (rank = %d):\n", root_comm);
		print_matrix(local_C, n, n);
		for (int i = 1; i < world_size; i++) {
			coords[0] = i / q; // row index
			coords[1] = i % q; // column index
			MPI_Cart_rank(comm, coords, &i_rank);
			MPI_Recv(tmp_C, n * n, MPI_FLOAT, i_rank, 0, comm, MPI_STATUS_IGNORE);
			printf("\nC%d%d (rank = %d):\n", coords[0], coords[1], i_rank);
			print_matrix(tmp_C, n, n);
		}
	}
	else {
		MPI_Send(local_C, n * n, MPI_FLOAT, root_comm, 0, comm);
	}

	// -> Release resources
	free(local_A);
	free(local_B);
	free(local_C);
	free(tmp_A);
	free(tmp_B);
	free(tmp_C);

	// -> OK!!!
	MPI_Finalize();
	return EXIT_SUCCESS;
}
