#include <cupy/complex.cuh>
#define PI 3.14159265359
extern "C" __global__
void multicorr_row_kernel(
	complex<float> *ar,
	const float *xyShifts,
	const long long N_pts,
	const long long image_size_x,
	const long long image_size_y,
	const long long upsample_factor) {
	/*
	Fill in the entries of the multicorr row kernel.
	Inputs:
		ar (complex<float>* / np.complex64):	Array of size N_pts x kernel_width * image_size[0]
				to hold the row kernels
		xyShifts (const float* / np.float32): (N_pts x 2) array of center points to build kernels for
		N_pts (const long long/Python int) number of center points we are
				building kernels for
		image_size_x (const long long/Python int): x size of the correlation image
		image_size_y (const long long/Python int): y size of correlation image
		upsample_factor (const long long/Python int): note, kernel_width = ceil(1.5*upsample_factor)
	*/
	int kernel_size = ceil(1.5 * upsample_factor);

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// Which kernel in the stack (first index of ar)
	int kernel_idx = tid / (kernel_size * image_size_x);
	// Which row in the kernel (second index of ar)
	int row_idx = (tid - (kernel_size*image_size_x)*kernel_idx) / image_size_x;
	// Which column in the kernel (last index of ar)
	int col_idx = (tid - (kernel_size*image_size_x)*kernel_idx - image_size_x*row_idx) % image_size_x;

	complex<float> prefactor = complex<float>(2.0 * PI,-1.0) / (image_size_y * upsample_factor);

	// Now do the actual calculation
	if (tid < N_pts * image_size_y * kernel_size) {
		float columnEntry = 0. ; //TODO

		// np.arange(numColumns) - xyShift[idx,1]
		float rowEntry = (float) col_idx - xyShifts[kernel_idx*2 + 1];

		ar[tid] = exp(prefactor * columnEntry * rowEntry); // Do I have to cast these explicitly?
	}

}