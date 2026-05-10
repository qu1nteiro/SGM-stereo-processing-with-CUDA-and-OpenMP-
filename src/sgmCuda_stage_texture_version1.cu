
// Based on CUDA SDK template from NVIDIA
// sgm algorithm adapted from http://lunokhod.org/?p=1403

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

#include <limits>
#include <algorithm>

// includes, project
#include <helper_cuda.h>
#include <helper_image.h>

#define MMAX_BRIGHTNESS 255

#define PENALTY1 15
#define PENALTY2 100

#define COSTS(i,j,d)              costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define ACCUMULATED_COSTS(i,j,d)  accumulated_costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define LEFT_IMAGE(i,j)           left_image[(i)+(j)*nx]
#define RIGHT_IMAGE(i,j)          right_image[(i)+(j)*nx]
#define DISP_IMAGE(i,j)           disp_image[(i)+(j)*nx]

#define MMAX(a,b) (((a)>(b))?(a):(b))
#define MMIN(a,b) (((a)<(b))?(a):(b))

// pixel base type
// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

/* ============================================================
 * Texture references for the two input images.
 *
 * Declared at file scope so that all device kernels can access
 * them. Texture memory is read-only and backed by a dedicated
 * cache optimised for the spatial access patterns used in SGM
 * (each thread reads a pixel and its immediate neighbours).
 *
 * The images are bound to these references in sgmDevice() after
 * the data has been copied to VRAM, and unbound before freeing.
 * ============================================================ */
texture<int, 1, cudaReadModeElementType> tex_left;
texture<int, 1, cudaReadModeElementType> tex_right;

/* function headers */

void determine_costs(const pixel_t *left_image, const pixel_t *right_image, int *costs, 
                     const int nx, const int ny, const int disp_range);

void evaluate_path( const int *prior, const int* local,
                    int path_intensity_gradient, int *curr_cost, 
                    const int nx, const int ny, const int disp_range );

void iterate_direction_dirxpos(const int dirx, const pixel_t *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirypos(const int diry, const pixel_t *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirxneg(const int dirx, const pixel_t *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_diryneg(const int diry, const pixel_t *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction( const int dirx, const int diry, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) ;

void inplace_sum_views( int * im1, const int * im2, 
                        const int nx, const int ny, const int disp_range ) ;

int find_min_index( const int *v, const int dist_range ) ;

void create_disparity_view( const int *accumulated_costs , pixel_t * disp_image, int nx, int ny) ;

void sgmHost(   const pixel_t *h_leftIm, const pixel_t *h_rightIm, 
                pixel_t *h_dispIm, 
                const int w, const int h, const int disp_range );

void sgmDevice( const pixel_t *h_leftIm, const pixel_t *h_rightIm, 
                pixel_t *h_dispImD, 
                const int w, const int h, const int disp_range );

void usage(char *command);


/* functions code */

void determine_costs(const pixel_t *left_image, const pixel_t *right_image, int *costs, 
                     const int nx, const int ny, const int disp_range)
{
  std::fill(costs, costs+nx*ny*disp_range, 255u);

  for ( int j = 0; j < ny; j++ ) {
      for ( int d = 0; d < disp_range; d++ ) {
          for ( int i = d; i < nx; i++ ) {
              COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
        }
      }
  }
}

void iterate_direction_dirxpos(const int dirx, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int j = 0; j < HEIGHT; j++ ) {
          for ( int i = 0; i < WIDTH; i++ ) {
              if(i==0) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)) ,
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range);
              }
          }
      }
}

void iterate_direction_dirypos(const int diry, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = 0; j < HEIGHT; j++ ) {
              if(j==0) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}

void iterate_direction_dirxneg(const int dirx, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int j = 0; j < HEIGHT; j++ ) {
          for ( int i = WIDTH-1; i >= 0; i-- ) {
              if(i==WIDTH-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}

void iterate_direction_diryneg(const int diry, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = HEIGHT-1; j >= 0; j-- ) {
              if(j==HEIGHT-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                           &COSTS(i,j,0),
                           abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                           &ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range);
             }
         }
      }
}
 
void iterate_direction( const int dirx, const int diry, const pixel_t *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    // Walk along the edges in a clockwise fashion
    if ( dirx > 0 ) {
      // LEFT MOST EDGE
      // Process every pixel along this edge
      iterate_direction_dirxpos(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( diry > 0 ) {
      // TOP MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the top left most pixel
      iterate_direction_dirypos(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( dirx < 0 ) {
      // RIGHT MOST EDGE
      // Process every pixel along this edge only if diry ==
      // 0. Otherwise skip the top right most pixel
      iterate_direction_dirxneg(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( diry < 0 ) {
      // BOTTOM MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the bottom left and bottom right pixel
      iterate_direction_diryneg(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
}

// ADD two cost images 
void inplace_sum_views( int * im1, const int * im2, 
                        const int nx, const int ny, const int disp_range ) 
{
    int *im1_init = im1;
    while ( im1 != (im1_init + (nx*ny*disp_range)) ) {
      *im1 += *im2;
      im1++;
      im2++;
    }
}

int find_min_index( const int *v, const int disp_range ) 
{
    int min = std::numeric_limits<int>::max();
    int minind = -1;
    for (int d=0; d < disp_range; d++) {
         if(v[d]<min) {
              min = v[d];
              minind = d;
         }
    }
    return minind;
}

void evaluate_path(const int *prior, const int *local,
                   int path_intensity_gradient, int *curr_cost , 
                   const int nx, const int ny, const int disp_range) 
{
  memcpy(curr_cost, local, sizeof(int)*disp_range);

  for ( int d = 0; d < disp_range; d++ ) {
    int e_smooth = std::numeric_limits<int>::max();
    for ( int d_p = 0; d_p < disp_range; d_p++ ) {
      if ( d_p - d == 0 ) {
        // No penality
        e_smooth = MMIN(e_smooth,prior[d_p]);
      } else if ( abs(d_p - d) == 1 ) {
        // Small penality
        e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
      } else {
        // Large penality
        e_smooth =
          MMIN(e_smooth,prior[d_p] +
                   MMAX(PENALTY1,
                            path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
      }
    }
    curr_cost[d] += e_smooth;
  }

  int min = std::numeric_limits<int>::max();
  for ( int d = 0; d < disp_range; d++ ) {
        if (prior[d]<min) min=prior[d];
  }
  for ( int d = 0; d < disp_range; d++ ) {
        curr_cost[d]-=min;
  }
}

void create_disparity_view( const int *accumulated_costs , pixel_t * disp_image, 
                            const int nx, const int ny, const int disp_range) 
{
  for ( int j = 0; j < ny; j++ ) {
    for ( int i = 0; i < nx; i++ ) {
      DISP_IMAGE(i,j) =
        4 * find_min_index( &ACCUMULATED_COSTS(i,j,0), disp_range );
    }
  }
}


 

/*
 * Links:
 * http://www.dlr.de/rmc/rm/en/desktopdefault.aspx/tabid-9389/16104_read-39811/
 * http://lunokhod.org/?p=1356
 */

/* ============================================================
 * CUDA KERNELS — Parallelisation step 1: determine_costs
 *
 * Strategy: assign one GPU thread to each (j, d) pair, mirroring the
 * two outermost loops of the sequential version. Each thread then walks
 * the valid column range i = [d, nx) on its own — that inner loop has
 * no cross-thread dependency, so this decomposition is race-free.
 *
 * In the OpenMP version we only parallelised the j-loop. Here we go
 * one level further and also spread work across the d dimension, keeping
 * threads busy even for small images.
 * ============================================================ */

/* ------------------------------------------------------------
 * kernel_fill
 *
 * Fills every element of an int array with a constant value.
 * We use this to initialise d_costs to 255 before the main cost
 * kernel runs: pixels where i < d must stay at 255 (no valid
 * right-image match), and the cost kernel only writes entries
 * where i >= d, so those sentinel values are never overwritten.
 *
 * One thread per array element; standard 1-D grid layout.
 * ------------------------------------------------------------ */
__global__ void kernel_fill(int *arr, int value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    arr[idx] = value;
}

/* ------------------------------------------------------------
 * kernel_determine_costs
 *
 * Each thread owns one (j, d) pair.
 *   j — image row  [0, ny)
 *   d — disparity  [0, disp_range)
 *
 * For every valid column i >= d the thread writes the absolute
 * pixel difference between the left image at (i, j) and the right
 * image shifted left by d pixels, i.e. at (i-d, j). This is the
 * Birchfield-Tomasi cost used by the SGM algorithm.
 *
 * Entries with i < d remain at the sentinel value of 255 set by
 * kernel_fill — no right-image pixel exists that far to the left.
 * ------------------------------------------------------------ */
__global__ void kernel_determine_costs(const pixel_t *left_image,
                                       const pixel_t *right_image,
                                       int *costs,
                                       int nx, int ny, int disp_range)
{
    /* Map this thread to its (j, d) coordinates */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    /* Discard threads that land outside the valid image/disparity range */
    if (j >= ny || d >= disp_range) return;

    /* Walk every column that has a valid right-image counterpart */
    for (int i = d; i < nx; i++) {
        costs[i * disp_range + j * nx * disp_range + d] =
            abs(left_image[i + j * nx] - right_image[(i - d) + j * nx]);
    }
}

/* ============================================================
 * CUDA KERNELS — Parallelisation step 2: iterate_direction
 *
 * Strategy: mirror the OpenMP parallelisation — one thread per
 * independent path. For horizontal directions (left↔right) each
 * row is an independent path, so one thread handles one row.
 * For vertical directions (top↔bottom) each column is independent,
 * so one thread handles one column.
 *
 * The sequential dependency along each path (pixel i depends on
 * pixel i-1) cannot be broken; that inner loop stays sequential
 * inside each thread, exactly as it did in the OpenMP version.
 *
 * evaluate_path() is a CPU function and cannot be called from
 * device code. We define device_evaluate_path() below with
 * identical logic, replacing memcpy with a manual loop and
 * std::numeric_limits<int>::max() with INT_MAX.
 * ============================================================ */

/* ------------------------------------------------------------
 * device_evaluate_path
 *
 * Identical logic to the host-side evaluate_path(). Marked
 * __device__ so that the four direction kernels can call it.
 *
 * Key differences from the CPU version:
 *   - memcpy replaced by a plain loop (not available on device)
 *   - std::numeric_limits<int>::max() replaced by INT_MAX
 * ------------------------------------------------------------ */
__device__ void device_evaluate_path(const int *prior, const int *local,
                                     int path_intensity_gradient,
                                     int *curr_cost, int disp_range)
{
    /* Copy local costs into curr_cost (memcpy is not available in device code) */
    for (int d = 0; d < disp_range; d++)
        curr_cost[d] = local[d];

    /* For each disparity d, find the minimum smoothed cost from the prior step */
    for (int d = 0; d < disp_range; d++) {
        int e_smooth = INT_MAX;
        for (int d_p = 0; d_p < disp_range; d_p++) {
            if (d_p == d) {
                /* Same disparity: no penalty */
                e_smooth = MMIN(e_smooth, prior[d_p]);
            } else if (abs(d_p - d) == 1) {
                /* Neighbouring disparity: small penalty */
                e_smooth = MMIN(e_smooth, prior[d_p] + PENALTY1);
            } else {
                /* Distant disparity: large penalty, modulated by intensity gradient */
                e_smooth = MMIN(e_smooth, prior[d_p] +
                           MMAX(PENALTY1, path_intensity_gradient ?
                                PENALTY2 / path_intensity_gradient : PENALTY2));
            }
        }
        curr_cost[d] += e_smooth;
    }

    /* Subtract the minimum prior cost to keep values bounded (SGM normalisation) */
    int min_prior = INT_MAX;
    for (int d = 0; d < disp_range; d++)
        if (prior[d] < min_prior) min_prior = prior[d];
    for (int d = 0; d < disp_range; d++)
        curr_cost[d] -= min_prior;
}

/* ------------------------------------------------------------
 * kernel_iterate_dirxpos  (direction: left to right, dirx = +1)
 *
 * One thread per row j. The thread initialises the left border
 * pixel and then walks right, each step depending on the previous.
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirxpos(const pixel_t *left_image,
                                       const int *costs,
                                       int *accumulated_costs,
                                       int nx, int ny, int disp_range)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    /* Left border: no prior exists, cost is taken directly */
    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(0, j, d) += COSTS(0, j, d);

    /* Walk left to right — sequential within the thread */
    for (int i = 1; i < nx; i++) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i-1, j, 0),
            &COSTS(i, j, 0),
            abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i-1, j)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_dirxneg  (direction: right to left, dirx = -1)
 *
 * One thread per row j. Starts at the right border and walks left.
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirxneg(const pixel_t *left_image,
                                       const int *costs,
                                       int *accumulated_costs,
                                       int nx, int ny, int disp_range)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    /* Right border: no prior exists, cost is taken directly */
    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(nx-1, j, d) += COSTS(nx-1, j, d);

    /* Walk right to left — sequential within the thread */
    for (int i = nx-2; i >= 0; i--) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i+1, j, 0),
            &COSTS(i, j, 0),
            abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i+1, j)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_dirypos  (direction: top to bottom, diry = +1)
 *
 * One thread per column i. Starts at the top border and walks down.
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirypos(const pixel_t *left_image,
                                       const int *costs,
                                       int *accumulated_costs,
                                       int nx, int ny, int disp_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    /* Top border: no prior exists, cost is taken directly */
    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(i, 0, d) += COSTS(i, 0, d);

    /* Walk top to bottom — sequential within the thread */
    for (int j = 1; j < ny; j++) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i, j-1, 0),
            &COSTS(i, j, 0),
            abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i, j-1)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_diryneg  (direction: bottom to top, diry = -1)
 *
 * One thread per column i. Starts at the bottom border and walks up.
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_diryneg(const pixel_t *left_image,
                                       const int *costs,
                                       int *accumulated_costs,
                                       int nx, int ny, int disp_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    /* Bottom border: no prior exists, cost is taken directly */
    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(i, ny-1, d) += COSTS(i, ny-1, d);

    /* Walk bottom to top — sequential within the thread */
    for (int j = ny-2; j >= 0; j--) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i, j+1, 0),
            &COSTS(i, j, 0),
            abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i, j+1)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ============================================================
 * CUDA KERNELS — Texture memory variants
 *
 * These kernels replace the global memory versions in sgmDevice()
 * when texture memory is used. The logic is identical; the only
 * difference is how the two input images are read:
 *
 *   Global memory:   left_image[i + j * nx]
 *   Texture memory:  tex1Dfetch(tex_left, i + j * nx)
 *
 * tex1Dfetch() routes the read through the texture cache, which
 * is separate from the L1/L2 caches used by global memory. For
 * access patterns with spatial locality (neighbouring pixels),
 * this can reduce memory latency significantly.
 *
 * Placed after Step 2 so that device_evaluate_path() — defined
 * there — is already visible when these kernels are compiled.
 *
 * Because the texture references are global, these kernels do NOT
 * receive the image arrays as parameters — they read directly
 * from tex_left and tex_right.
 * ============================================================ */

/* ------------------------------------------------------------
 * kernel_determine_costs_tex
 *
 * Same as kernel_determine_costs but reads both images through
 * the texture cache instead of global memory.
 * ------------------------------------------------------------ */
__global__ void kernel_determine_costs_tex(int *costs,
                                           int nx, int ny, int disp_range)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= ny || d >= disp_range) return;

    for (int i = d; i < nx; i++) {
        costs[i * disp_range + j * nx * disp_range + d] =
            abs(tex1Dfetch(tex_left,  i       + j * nx) -
                tex1Dfetch(tex_right, (i - d) + j * nx));
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_dirxpos_tex  (left to right, dirx = +1)
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirxpos_tex(const int *costs,
                                           int *accumulated_costs,
                                           int nx, int ny, int disp_range)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(0, j, d) += COSTS(0, j, d);

    for (int i = 1; i < nx; i++) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i-1, j, 0),
            &COSTS(i, j, 0),
            abs(tex1Dfetch(tex_left, i     + j * nx) -
                tex1Dfetch(tex_left, (i-1) + j * nx)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_dirxneg_tex  (right to left, dirx = -1)
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirxneg_tex(const int *costs,
                                           int *accumulated_costs,
                                           int nx, int ny, int disp_range)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(nx-1, j, d) += COSTS(nx-1, j, d);

    for (int i = nx-2; i >= 0; i--) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i+1, j, 0),
            &COSTS(i, j, 0),
            abs(tex1Dfetch(tex_left, i     + j * nx) -
                tex1Dfetch(tex_left, (i+1) + j * nx)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_dirypos_tex  (top to bottom, diry = +1)
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_dirypos_tex(const int *costs,
                                           int *accumulated_costs,
                                           int nx, int ny, int disp_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(i, 0, d) += COSTS(i, 0, d);

    for (int j = 1; j < ny; j++) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i, j-1, 0),
            &COSTS(i, j, 0),
            abs(tex1Dfetch(tex_left, i + j     * nx) -
                tex1Dfetch(tex_left, i + (j-1) * nx)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ------------------------------------------------------------
 * kernel_iterate_diryneg_tex  (bottom to top, diry = -1)
 * ------------------------------------------------------------ */
__global__ void kernel_iterate_diryneg_tex(const int *costs,
                                           int *accumulated_costs,
                                           int nx, int ny, int disp_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    for (int d = 0; d < disp_range; d++)
        ACCUMULATED_COSTS(i, ny-1, d) += COSTS(i, ny-1, d);

    for (int j = ny-2; j >= 0; j--) {
        device_evaluate_path(
            &ACCUMULATED_COSTS(i, j+1, 0),
            &COSTS(i, j, 0),
            abs(tex1Dfetch(tex_left, i + j     * nx) -
                tex1Dfetch(tex_left, i + (j+1) * nx)),
            &ACCUMULATED_COSTS(i, j, 0),
            disp_range);
    }
}

/* ============================================================
 * END OF CUDA KERNELS — Texture memory variants
 * ============================================================ */

/* ============================================================
 * CUDA KERNELS — Parallelisation step 3: inplace_sum_views
 *
 * The original function walks two arrays element by element and
 * adds them in place: im1[k] += im2[k]. Every element is fully
 * independent, so we assign one thread per element and let all
 * of them run in parallel — no dependencies, no shared state.
 *
 * This replaces the four CPU round-trips from Step 2: d_dir_accumulated
 * now stays on the GPU and is summed directly into d_total_accumulated
 * without ever leaving VRAM.
 * ============================================================ */

/* ------------------------------------------------------------
 * kernel_inplace_sum_views
 *
 * Adds every element of im2 into im1 in place: im1[k] += im2[k].
 * Called once per direction, with:
 *   im1 = d_total_accumulated  (accumulates across all directions)
 *   im2 = d_dir_accumulated    (result of the most recent direction)
 *
 * Standard 1-D grid: one thread per array element.
 * ------------------------------------------------------------ */
__global__ void kernel_inplace_sum_views(int *im1, const int *im2, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    im1[k] += im2[k];
}

/* ============================================================
 * CUDA KERNELS — Parallelisation step 4: create_disparity_view
 *
 * Each pixel (i, j) independently finds the disparity value that
 * minimises its accumulated cost. There are no dependencies between
 * pixels, so we assign one thread per pixel using a 2-D grid — the
 * natural fit for image data.
 *
 * find_min_index() is a CPU helper; its logic is inlined here
 * directly inside the kernel rather than wrapping it in a separate
 * __device__ function, since it is a short loop and called only once
 * per thread.
 * ============================================================ */

/* ------------------------------------------------------------
 * kernel_create_disparity_view
 *
 * Each thread owns one pixel (i, j). It scans the disp_range
 * accumulated cost values for that pixel, finds the index of the
 * minimum, multiplies by 4 (the SGM display convention), and writes
 * the result into the disparity image.
 *
 * Grid layout: 2-D, one thread per pixel.
 *   blockDim = (16, 16)  →  256 threads per block
 *   gridDim.x = ceil(nx / 16)
 *   gridDim.y = ceil(ny / 16)
 * ------------------------------------------------------------ */
__global__ void kernel_create_disparity_view(const int *accumulated_costs,
                                             pixel_t *disp_image,
                                             int nx, int ny, int disp_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    /* Find the disparity with the lowest accumulated cost for this pixel */
    int min_cost = INT_MAX;
    int min_d    = 0;
    for (int d = 0; d < disp_range; d++) {
        int cost = accumulated_costs[i * disp_range + j * nx * disp_range + d];
        if (cost < min_cost) {
            min_cost = cost;
            min_d    = d;
        }
    }

    disp_image[i + j * nx] = 4 * min_d;
}

/* ============================================================
 * END OF CUDA KERNELS — Parallelisation step 4
 * ============================================================ */

// sgm code to run on the host
void sgmHost(   const pixel_t *h_leftIm, const pixel_t *h_rightIm, 
                pixel_t *h_dispIm, 
                const int w, const int h, const int disp_range)
{
    const int nx = w;
    const int ny = h;
 
  // Processing all costs. W*H*D. D= disp_range
  int *costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  if (costs == NULL) { 
        fprintf(stderr, "sgm_cuda:"
                " Failed memory allocation(s).\n");
        exit(1);
  }

  determine_costs(h_leftIm, h_rightIm, costs, nx, ny, disp_range);

  int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  if (accumulated_costs == NULL || dir_accumulated_costs == NULL) { 
        fprintf(stderr, "sgm_cuda:"
                " Failed memory allocation(s).\n");
        exit(1);
  }

  int dirx=0,diry=0;
  for(dirx=-1; dirx<2; dirx++) {
      if(dirx==0 && diry==0) continue;
      std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
      iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
      inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
  }
  dirx=0;
  for(diry=-1; diry<2; diry++) {
      if(dirx==0 && diry==0) continue;
      std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
      iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
      inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
  }

  free(costs);
  free(dir_accumulated_costs);

  create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );

  free(accumulated_costs);
}

// sgm code to run on the GPU
void sgmDevice( const pixel_t *h_leftIm, const pixel_t *h_rightIm, 
                pixel_t *h_dispImD, 
                const int w, const int h, const int disp_range )
{
    const int nx = w;
    const int ny = h;
    const int total_elems = nx * ny * disp_range;

    /* ----------------------------------------------------------------
     * Device memory pointers (prefix d_ = lives on GPU VRAM)
     * ---------------------------------------------------------------- */
    pixel_t *d_left             = NULL;
    pixel_t *d_right            = NULL;
    int     *d_costs            = NULL;
    int     *d_dir_accumulated  = NULL;
    int     *d_total_accumulated = NULL;

    /* ----------------------------------------------------------------
     * Allocate GPU memory
     * ---------------------------------------------------------------- */
    checkCudaErrors(cudaMalloc(&d_left,             nx * ny * sizeof(pixel_t)));
    checkCudaErrors(cudaMalloc(&d_right,            nx * ny * sizeof(pixel_t)));
    checkCudaErrors(cudaMalloc(&d_costs,            total_elems * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_dir_accumulated,  total_elems * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_total_accumulated, total_elems * sizeof(int)));

    /* d_total_accumulated starts at zero; directions are summed into it */
    checkCudaErrors(cudaMemset(d_total_accumulated, 0, total_elems * sizeof(int)));

    /* ----------------------------------------------------------------
     * Transfer input images from host RAM to device VRAM, then bind
     * them to the texture references so all kernels read through the
     * texture cache instead of going directly to global memory.
     * ---------------------------------------------------------------- */
    checkCudaErrors(cudaMemcpy(d_left,  h_leftIm,  nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_right, h_rightIm, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaBindTexture(NULL, tex_left,  d_left,  nx * ny * sizeof(pixel_t)));
    checkCudaErrors(cudaBindTexture(NULL, tex_right, d_right, nx * ny * sizeof(pixel_t)));

    /* ----------------------------------------------------------------
     * Step 1 - determine_costs (GPU, texture variant)
     * Images are read via tex1Dfetch() instead of pointer dereference.
     * ---------------------------------------------------------------- */
    {
        int threads = 256;
        int blocks  = (total_elems + threads - 1) / threads;
        kernel_fill<<<blocks, threads>>>(d_costs, 255, total_elems);
    }
    {
        dim3 block(16, 16);
        dim3 grid((ny         + block.x - 1) / block.x,
                  (disp_range + block.y - 1) / block.y);
        kernel_determine_costs_tex<<<grid, block>>>(d_costs, nx, ny, disp_range);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    /* ----------------------------------------------------------------
     * Steps 2 + 3 - iterate_direction (texture) + inplace_sum_views
     * ---------------------------------------------------------------- */
    int row_threads = 256;
    int row_blocks  = (ny + row_threads - 1) / row_threads;

    int col_threads = 256;
    int col_blocks  = (nx + col_threads - 1) / col_threads;

    int sum_threads = 256;
    int sum_blocks  = (total_elems + sum_threads - 1) / sum_threads;

    /* --- Direction: left to right (dirx = +1) --- */
    checkCudaErrors(cudaMemset(d_dir_accumulated, 0, total_elems * sizeof(int)));
    kernel_iterate_dirxpos_tex<<<row_blocks, row_threads>>>(d_costs, d_dir_accumulated, nx, ny, disp_range);
    kernel_inplace_sum_views<<<sum_blocks, sum_threads>>>(d_total_accumulated, d_dir_accumulated, total_elems);

    /* --- Direction: right to left (dirx = -1) --- */
    checkCudaErrors(cudaMemset(d_dir_accumulated, 0, total_elems * sizeof(int)));
    kernel_iterate_dirxneg_tex<<<row_blocks, row_threads>>>(d_costs, d_dir_accumulated, nx, ny, disp_range);
    kernel_inplace_sum_views<<<sum_blocks, sum_threads>>>(d_total_accumulated, d_dir_accumulated, total_elems);

    /* --- Direction: top to bottom (diry = +1) --- */
    checkCudaErrors(cudaMemset(d_dir_accumulated, 0, total_elems * sizeof(int)));
    kernel_iterate_dirypos_tex<<<col_blocks, col_threads>>>(d_costs, d_dir_accumulated, nx, ny, disp_range);
    kernel_inplace_sum_views<<<sum_blocks, sum_threads>>>(d_total_accumulated, d_dir_accumulated, total_elems);

    /* --- Direction: bottom to top (diry = -1) --- */
    checkCudaErrors(cudaMemset(d_dir_accumulated, 0, total_elems * sizeof(int)));
    kernel_iterate_diryneg_tex<<<col_blocks, col_threads>>>(d_costs, d_dir_accumulated, nx, ny, disp_range);
    kernel_inplace_sum_views<<<sum_blocks, sum_threads>>>(d_total_accumulated, d_dir_accumulated, total_elems);

    checkCudaErrors(cudaDeviceSynchronize());

    /* ----------------------------------------------------------------
     * Step 4 - create_disparity_view (GPU)
     *
     * d_total_accumulated stays in VRAM — no round-trip to the CPU.
     * One thread per pixel (i, j); 2-D grid mirrors the image layout.
     *
     * The only cudaMemcpy left in the entire pipeline is this final
     * copy of the disparity image back to the host — unavoidable,
     * since the caller expects the result in h_dispImD (host memory).
     * That copy is nx*ny ints (~675 KB for the test image), far smaller
     * than the cost-volume copies we eliminated in earlier stages.
     * ---------------------------------------------------------------- */
    pixel_t *d_disp = NULL;
    checkCudaErrors(cudaMalloc(&d_disp, nx * ny * sizeof(pixel_t)));

    {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y);
        kernel_create_disparity_view<<<grid, block>>>(d_total_accumulated,
                                                      d_disp,
                                                      nx, ny, disp_range);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    /* Copy the disparity image back to the host (result expected in RAM) */
    checkCudaErrors(cudaMemcpy(h_dispImD, d_disp,
                               nx * ny * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    /* ----------------------------------------------------------------
     * Cleanup: unbind textures and free all device allocations.
     * Textures must be unbound before the underlying device memory
     * is freed — accessing a bound texture after cudaFree is undefined.
     * ---------------------------------------------------------------- */
    cudaUnbindTexture(tex_left);
    cudaUnbindTexture(tex_right);

    checkCudaErrors(cudaFree(d_left));
    checkCudaErrors(cudaFree(d_right));
    checkCudaErrors(cudaFree(d_costs));
    checkCudaErrors(cudaFree(d_dir_accumulated));
    checkCudaErrors(cudaFree(d_total_accumulated));
    checkCudaErrors(cudaFree(d_disp));
}

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-d device] [-l leftimage] [-r rightimage] [-o dev_dispimage] [-t host_dispimage] [-p disprange] \n",command);
}

// main
int main( int argc, char** argv) 
{

    // default command line options
    int deviceId = 0;
    int disp_range = 32;
    char *leftIn      =(char *)"lbull.pgm",
         *rightIn     =(char *)"rbull.pgm",
         *fileOut     =(char *)"d_dbull_cuda.pgm",
         *referenceOut=(char *)"h_dbull_cuda.pgm";

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:l:o:r:t:p:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':  // device
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'l': // left image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                leftIn = strdup(optarg);
                break;
            case 'r': // right image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                rightIn = strdup(optarg);
                break;
            case 'o': // output image (from device) filename 
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 't': // output image (from host) filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'p': // disp_range
                if(sscanf(optarg,"%d",&disp_range)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'h': // help
                usage(argv[0]);
                exit(0);
                break;

        }
    }

    if(optind < argc) {
        fprintf(stderr,"Error in arguments\n");
        usage(argv[0]);
        exit(1);
    }

    // select cuda device
    checkCudaErrors( cudaSetDevice( deviceId ) );
    
    // create events to measure host sgm time and device sgm time
    cudaEvent_t startH, stopH, startD, stopD;
    checkCudaErrors(cudaEventCreate(&startH));
    checkCudaErrors(cudaEventCreate(&stopH));
    checkCudaErrors(cudaEventCreate(&startD));
    checkCudaErrors(cudaEventCreate(&stopD));

    // allocate host memory
    int* h_ldata=NULL;
    int* h_rdata=NULL;
    unsigned int h,w;

    //load left pgm
    if (sdkLoadPGM<pixel_t>(leftIn, &h_ldata, &w, &h) != true) {
        printf("Failed to load image file: %s\n", leftIn);
        exit(1);
    }
    //load right pgm
    if (sdkLoadPGM<pixel_t>(rightIn, &h_rdata, &w, &h) != true) {
        printf("Failed to load image file: %s\n", rightIn);
        exit(1);
    }

    // allocate mem for the result on host side
    int* h_odata = (int*) malloc( h*w*sizeof(int));
    int* reference = (int*) malloc( h*w*sizeof(int));
 
    // sgm at host
    checkCudaErrors(cudaEventRecord( startH, 0 ));
    sgmHost(h_ldata, h_rdata, reference, w, h, disp_range);   
    checkCudaErrors(cudaEventRecord( stopH, 0 )); 
    checkCudaErrors(cudaEventSynchronize( stopH ));

    // sgm at GPU
    checkCudaErrors(cudaEventRecord( startD, 0 ));
    sgmDevice(h_ldata, h_rdata, h_odata, w, h, disp_range);   
    checkCudaErrors(cudaEventRecord( stopD, 0 )); 
    checkCudaErrors(cudaEventSynchronize( stopD ));
    
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    float timeH, timeD;
    checkCudaErrors(cudaEventElapsedTime( &timeH, startH, stopH ));
    printf( "Host processing time: %f (ms)\n", timeH);
    checkCudaErrors(cudaEventElapsedTime( &timeD, startD, stopD ));
    printf( "Device processing time: %f (ms)\n", timeD);

    // save output images
    if (sdkSavePGM<pixel_t>(referenceOut, reference, w, h) != true) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (sdkSavePGM<pixel_t>(fileOut, h_odata, w, h) != true) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    free(h_ldata);
    free(h_rdata);
    free(h_odata);
    free(reference);

    checkCudaErrors( cudaDeviceReset() );
}
