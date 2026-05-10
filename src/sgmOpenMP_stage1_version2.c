
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
#include <sys/time.h>

#include <limits>
#include <algorithm>
#include <omp.h>

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

void sgmOpenMP( const pixel_t *h_leftIm, const pixel_t *h_rightIm, 
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

// sgmOpenMP — Version 2: determine_costs and iterate_direction run in parallel.
// For each scanning direction, the outer loop (rows for horizontal, columns for
// vertical) is parallelised. The inner loop stays sequential because each pixel
// depends on the previous one along the path.
// inplace_sum_views and create_disparity_view still run serially.
void sgmOpenMP( const pixel_t *left_image, const pixel_t *right_image,
                pixel_t *disp_image, 
                const int w, const int h, const int disp_range )
{
     const int nx = w;
     const int ny = h;

     int *costs = (int*) calloc(nx * ny * disp_range, sizeof(int));
     if (costs == NULL) {
	     fprintf(stderr, "smgOpenMP: failed memory allocation (costs).\n");
	     exit(1);
    }	     

    std::fill(costs, costs + nx * ny * disp_range, 255u); 
    
    // Rows are independent: each costs[i][j][d] is written exactly once and reads
    // only from the read-only input images, so there are no data races.
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int d = 0; d < disp_range; d++) {
		    for (int i = d; i < nx; i++) {
			    COSTS(i, j, d) = abs( LEFT_IMAGE(i, j) - RIGHT_IMAGE(i -d, j) );
		    }
	    }
    }

      int *total_accumulated_costs = (int *) calloc(nx * ny * disp_range, sizeof(int));
      int *dir_accumulated_costs   = (int *) calloc(nx * ny * disp_range, sizeof(int));
      if (total_accumulated_costs == NULL || dir_accumulated_costs == NULL) {
          fprintf(stderr, "sgmOpenMP: failed memory allocation (accumulated costs).\n");
          exit(1);
      }
 
      int dirx = 0, diry = 0;
 
      /* Horizontal directions: dirx = -1 (right to left) and dirx = +1 (left to right) */
      for (dirx = -1; dirx < 2; dirx++) {
          if (dirx == 0 && diry == 0) continue;
 
          std::fill(dir_accumulated_costs, dir_accumulated_costs + nx * ny * disp_range, 0);
          int *accumulated_costs = dir_accumulated_costs;

          if (dirx > 0) {
              // Each row is an independent path; the inner i-loop is sequential (i depends on i-1).
              #pragma omp parallel for
              for (int j = 0; j < ny; j++) {
                  for (int i = 0; i < nx; i++) {
                      if (i == 0) {
                          for (int d = 0; d < disp_range; d++)
                              ACCUMULATED_COSTS(0, j, d) += COSTS(0, j, d);
                      } else {
                          evaluate_path( &ACCUMULATED_COSTS(i - dirx, j, 0),
                                         &COSTS(i, j, 0),
                                         abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i - dirx, j)),
                                         &ACCUMULATED_COSTS(i, j, 0),
                                         nx, ny, disp_range );
                      }
                  }
              }
          } else {
              // Each row is an independent path; the inner i-loop is sequential (i depends on i-1).
              #pragma omp parallel for
              for (int j = 0; j < ny; j++) {
                  for (int i = nx - 1; i >= 0; i--) {
                      if (i == nx - 1) {
                          for (int d = 0; d < disp_range; d++)
                              ACCUMULATED_COSTS(nx - 1, j, d) += COSTS(nx - 1, j, d);
                      } else {
                          evaluate_path( &ACCUMULATED_COSTS(i - dirx, j, 0),
                                         &COSTS(i, j, 0),
                                         abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i - dirx, j)),
                                         &ACCUMULATED_COSTS(i, j, 0),
                                         nx, ny, disp_range );
                      }
                  }
              }
          }
 
          inplace_sum_views(total_accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
      }
 
      /* Vertical directions: diry = -1 (bottom to top) and diry = +1 (top to bottom) */
      dirx = 0;
      for (diry = -1; diry < 2; diry++) {
          if (dirx == 0 && diry == 0) continue;
 
          std::fill(dir_accumulated_costs, dir_accumulated_costs + nx * ny * disp_range, 0);
          int *accumulated_costs = dir_accumulated_costs;

          if (diry > 0) {
              // Each column is an independent path; the inner j-loop is sequential (j depends on j-1).
              #pragma omp parallel for
              for (int i = 0; i < nx; i++) {
                  for (int j = 0; j < ny; j++) {
                      if (j == 0) {
                          for (int d = 0; d < disp_range; d++)
                              ACCUMULATED_COSTS(i, 0, d) += COSTS(i, 0, d);
                      } else {
                          evaluate_path( &ACCUMULATED_COSTS(i, j - diry, 0),
                                         &COSTS(i, j, 0),
                                         abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i, j - diry)),
                                         &ACCUMULATED_COSTS(i, j, 0),
                                         nx, ny, disp_range );
                      }
                  }
              }
          } else {
              // Each column is an independent path; the inner j-loop is sequential (j depends on j-1).
              #pragma omp parallel for
              for (int i = 0; i < nx; i++) {
                  for (int j = ny - 1; j >= 0; j--) {
                      if (j == ny - 1) {
                          for (int d = 0; d < disp_range; d++)
                              ACCUMULATED_COSTS(i, ny - 1, d) += COSTS(i, ny - 1, d);
                      } else {
                          evaluate_path( &ACCUMULATED_COSTS(i, j - diry, 0),
                                         &COSTS(i, j, 0),
                                         abs(LEFT_IMAGE(i, j) - LEFT_IMAGE(i, j - diry)),
                                         &ACCUMULATED_COSTS(i, j, 0),
                                         nx, ny, disp_range );
                      }
                  }
              }
          }
 
          inplace_sum_views(total_accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
      }

    free(costs);
    free(dir_accumulated_costs);

    create_disparity_view(total_accumulated_costs, disp_image, nx, ny, disp_range );

    free(total_accumulated_costs);
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
         *fileOut     =(char *)"d_dbull_openmp.pgm",
         *referenceOut=(char *)"h_dbull_openmp.pgm";

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
    struct timeval start, end;
    gettimeofday(&start, NULL);

    sgmHost(h_ldata, h_rdata, reference, w, h, disp_range);   

    gettimeofday(&end, NULL);

    struct timeval startMP, endMP;
    gettimeofday(&startMP, NULL);

    // sgm with OpenMP
    sgmOpenMP(h_ldata, h_rdata, h_odata, w, h, disp_range);   

    gettimeofday(&endMP, NULL);
    
    printf( "Host processing time: %f (ms)\n", (end.tv_sec-start.tv_sec)*1000.0 + ((double)(end.tv_usec - start.tv_usec))/1000.0);
    printf( "OpenMP processing time: %f (ms)\n", (endMP.tv_sec-startMP.tv_sec)*1000.0 + ((double)(endMP.tv_usec - startMP.tv_usec))/1000.0);

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
}
