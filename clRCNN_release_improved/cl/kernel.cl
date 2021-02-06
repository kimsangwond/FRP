__kernel void sum(__global const float* a_g, __global const float* b_g, __global float *res_g)
{
	int gid = get_global_id(0);
	res_g[gid] = a_g[gid] + b_g[gid];
}

// height_col, width_col: output blob width, height size
// n = input channel * height_col * width_col
// global work num: n
__kernel void im2col_kernel(const int n, __global const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    __global float* data_col) 
{
    for(int index = get_global_id(0); index < n; index += get_global_size(0) ) 
    { 
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        __global float* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        __global const float* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        
        for (int i = 0; i < kernel_h; ++i) 
        {
            for (int j = 0; j < kernel_w; ++j) 
            {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

// input blob shape: (num, channels, height, width)
// output blob shape: (num, channels, pooled_height, pooled_width)
// nthreads: size of output blob
// global work num: nthreads
__kernel void max_pool_kernel(const int nthreads,
    __global const float* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    __global float* const top_data) 
{
    for(int index = get_global_id(0); index < nthreads; index += get_global_size(0) )
    {
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % channels;
        const int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float maxval = -FLT_MAX;
        __global const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) 
        {
            for (int w = wstart; w < wend; ++w) 
            {
                int idx = h * width + w;
                if (bottom_slice[idx] > maxval) 
                {
                    maxval = bottom_slice[idx];
                }
            }
        }
        top_data[index] = maxval;
    }
}

// C = alpha*(AB) + beta C
// C: MxN, A: MxK, B: K,N
// global_id(0) : N, global_id(1) : M

__kernel void poor_matmul( int M, int N, int K, 
    float alpha, __global const float* A, __global const float* B,
    float beta, __global float* C )
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if( 0<=i && i< N && 0<=j && j <M)
    {
        float sum = 0.f;
        for(int k = 0 ; k  < K ; ++k)
        {
            sum += A[j*K + k] * B[k*N + i];
        }

        
        //if(beta != 0)
        //    C[j*N + i] = alpha*sum + beta*C[j*N+i];
        //else
            C[j*N + i] = alpha*sum;
    }
}
/*
#define BLOCK_SIZE 8

__kernel void poor_matmul( int M, int N, int K,
	const float alpha, __global float* A, __global float* B, const float beta, __global float* C)//, __global float* P,  int group, int num_input_ch 
{    
	// Block index
    int bx = get_group_id(1);
    int by = get_group_id(0);
 
    // Thread index
    int tx = get_local_id(1);
    int ty = get_local_id(0);
 
    // Index of the first sub-matrix of A processed 
    // by the block
	int aBegin = BLOCK_SIZE * by;    /////////////////////
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + M*(K - 1); 
 
    // Step size used to iterate through the 
    // sub-matrices of A
	int aStep  = BLOCK_SIZE *M;
 
    // Index of the first sub-matrix of B processed 
    // by the block
	int bBegin = K *BLOCK_SIZE * bx; ///////////////////////////
 
	//int bEnd   = bBegin + K - 1; 
 
 
    // Step size used to iterate through the 
    // sub-matrices of B
	int bStep  = BLOCK_SIZE;
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix

	int g = 0;
		
		float Csub=0.0;

		for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
		{

			// Declaration of the local memory array As 
			// used to store the sub-matrix of A
			__local float As[BLOCK_SIZE][BLOCK_SIZE];
 
			// Declaration of the local memory array Bs 
			// used to store the sub-matrix of B
			__local float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
			// Load the matrices from global memory
			// to local memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[  K*M*g+  a + M * tx + ty];                    
			Bs[ty][tx] = B[  N*K*g+  b + K * tx + ty];                         
 
			// Synchronize to make sure the matrices 
			// are loaded
			barrier(CLK_LOCAL_MEM_FENCE);
 
			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
			for (int k = 0; k < BLOCK_SIZE; ++k)
				Csub += As[ty][k] * Bs[k][tx];
		
			  // Synchronize to make sure tMt the preceding
			  // computation is done before loading two new
			 // sub-matrices of A and B in the next iteration
			 barrier(CLK_LOCAL_MEM_FENCE);
 
		}
		
 
		// Write the block sub-matrix to device memory;
		// each thread writes one element
		if(
			(bx*BLOCK_SIZE+tx<N) && (by*BLOCK_SIZE+ty<M)
	    )
		{
			int c = M * BLOCK_SIZE * bx + BLOCK_SIZE * by;                      
			C[ N*M*g + c + M * tx + ty] = Csub;  

		}	
}
*/
__kernel void poor_matmul2( int M, int N, int K, 
    float alpha, __global const float* A, __global const float* B,
    float beta, __global float* C )
{
    for(int index = get_global_id(0); index < M*N; index += get_global_size(0))
    {
        int i = index % N;
        int j = index / N;

        float sum = 0;
        for(int k = 0 ; k  < K ; ++k)
        {
            sum += A[j*K + k] * B[k*N + i];
        }

        //C[index] = alpha*sum + beta*C[index];
        C[index] = alpha*sum;
    }
}

// spartial_dim = width * height
// shape(bias) = (channel)
// n = output channel * output width * output height
__kernel void add_bias(int n, int channel, int spartial_dim, 
    __global const float* bias, __global float* activ)
{
    for(int index = get_global_id(0); index < n; index +=get_global_size(0))
    {
        int ch_idx = (index / spartial_dim)%channel;
        activ[index] += bias[ch_idx];
    }
}

__kernel void relu(int n, 
    __global const float* input, __global float* output)
{
    for(int index = get_global_id(0); index < n; index +=get_global_size(0))
    {
       output[index] = input[index] > 0 ? input[index]: 0;
    }
}


__kernel void square_kernel(const int n, 
    __global const float* input, __global float* temp)
{
    for(int index = get_global_id(0); index < n; index += get_global_size(0) )
    {
        temp[index] = pow(input[index], 2);
    }
}

// temp tensor: power(input, 2) 
// n: the number of input(output) elements
__kernel void lrn_inter_kernel(const int n, __global const float* input, 
    const int num, const int channels, const int height, const int width, 
    const int local_size, const float alpha, const float beta, 
    __global float* temp, __global float* output) 
{
    for(int index = get_global_id(0); index < n; index += get_global_size(0) )
    {
        const int pw = index % width;
        const int ph = (index / width) % height;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;
        const int loc_pad = (local_size-1)/2;

        int hstart = ph - loc_pad;
        int wstart = pw - loc_pad;
        const int hend = min(hstart + local_size, height);
        const int wend = min(wstart + local_size, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        
        float avg_val = 0;

        __global  float*  temp_slice =
            temp + (n * channels + c) * height * width;

        for (int h = hstart; h < hend; ++h) 
        {
            for (int w = wstart; w < wend; ++w) 
            {
                avg_val += temp_slice[h * width + w];
            }
        }

        output[index] = input[index] / pow(1 + avg_val*alpha/(local_size*local_size), beta);
    }
}

__kernel void scale_kernel(const int n, const float alpha,
    __global float* data)
{
    for(int index = get_global_id(0); index < n; index += get_global_size(0) )
    {
        data[index] *= alpha;
    }
}

// nthreads: output num
__kernel void roi_pool_kernel(const int nthreads,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    __global const float* bottom_data, __global const float* bottom_rois,
    __global float* top_data) 
{
    for(int index = get_global_id(0); index < nthreads; index += get_global_size(0) )
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = ((index / pooled_width) / pooled_height) % channels;
        int n = ((index / pooled_width) / pooled_height) / channels;

        __global const float* bottom_rois_slice = bottom_rois + n*5;
        int roi_batch_ind = (int)bottom_rois_slice[0];
        int roi_start_w = (int)round(bottom_rois_slice[1] * spatial_scale);
        int roi_start_h = (int)round(bottom_rois_slice[2] * spatial_scale);
        int roi_end_w = (int)round(bottom_rois_slice[3] * spatial_scale);
        int roi_end_h = (int)round(bottom_rois_slice[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(1, roi_end_w - roi_start_w + 1);
        int roi_height = max(1, roi_end_h - roi_start_h + 1);
        float bin_size_h = ((float)roi_height) / ((float)pooled_height);
        float bin_size_w = ((float)roi_width) / ((float)pooled_width);

        float ph_bin = ph * bin_size_h;
        float pw_bin = pw * bin_size_w;
        int hstart = (int)(floor( ph_bin ));
        int wstart = (int)(floor( pw_bin ));
        ph_bin = ((ph + 1)* bin_size_h)-0.001;
        pw_bin = ((pw + 1)* bin_size_w)-0.001;
        int hend = (int)(ceil( ph_bin ));
        int wend = (int)(ceil( pw_bin ));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
    
        bool is_empty = ((hend <= hstart) || (wend <= wstart));

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        
        __global const float* bottom_slice =
            bottom_data + (roi_batch_ind * channels + c) * height * width;
           
            
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                
                if (bottom_slice[bottom_index] > maxval) {
                    maxval = bottom_slice[bottom_index];
                }
            }
        }
        
        top_data[index] = maxval;
    }
}

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK sizeof(unsigned long) * 8

inline float devIoU(__global float const * const a, __local float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__kernel void nms_kernel(
    const int n_boxes, const float nms_overlap_thresh,
    __global const float *dev_boxes, 
    __global unsigned long *dev_mask) 
{
  const int row_start = get_group_id(1);
  const int col_start = get_group_id(0);

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const int col_size =
        min(n_boxes - col_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  __local float block_boxes[THREADS_PER_BLOCK * 5];

  int tid = get_local_id(0);
  if (tid< col_size) {
    block_boxes[tid * 5 + 0] =
        dev_boxes[(THREADS_PER_BLOCK * col_start + tid) * 5 + 0];
    block_boxes[tid * 5 + 1] =
        dev_boxes[(THREADS_PER_BLOCK * col_start + tid) * 5 + 1];
    block_boxes[tid * 5 + 2] =
        dev_boxes[(THREADS_PER_BLOCK * col_start + tid) * 5 + 2];
    block_boxes[tid * 5 + 3] =
        dev_boxes[(THREADS_PER_BLOCK * col_start + tid) * 5 + 3];
    block_boxes[tid * 5 + 4] =
        dev_boxes[(THREADS_PER_BLOCK * col_start + tid) * 5 + 4];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < row_size) {
    const int cur_box_idx = THREADS_PER_BLOCK * row_start + tid;
    __global const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= ((unsigned long)1) << i;  // OpenCL unsigned long = 64 bit..is it correct for T628?
      }
    }
    const int col_blocks = DIVUP(n_boxes, THREADS_PER_BLOCK);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}