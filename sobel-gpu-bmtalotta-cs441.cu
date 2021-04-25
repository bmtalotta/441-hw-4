/***********************************************************************
 * sobel-gpu-bmtalotta-cs441.cu
 *
 * Implements a Sobel filter on the image that is hard-coded in main.
 * You might add the image name as a command line option if you were
 * to use this more than as a one-off assignment.
 *
 * See https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
 * or https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
 * for info on how the filter is implemented.
 *
 * Compile/run with:  nvcc sobel-cpu.cu -lfreeimage
 *
 ***********************************************************************/
#include "FreeImage.h"
#include "stdio.h"
#include "math.h"

// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
__device__ int pixelIndex(int x, int y, int width)
{
  return (y*width + x);
}

// Returns the sobel value for pixel x,y
__global__ void sobel(int width, int height, char *pixels, int *arr)
{
  //y uses .x and x uses .y for this :(
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if(x > 0 && y > 0 && x < width - 1 && y < width - 1){
    int x00 = -1;  int x20 = 1;
    int x01 = -2;  int x21 = 2;
    int x02 = -1;  int x22 = 1;
    x00 *= pixels[pixelIndex(x-1,y-1,width)];
    x01 *= pixels[pixelIndex(x-1,y,width)];
    x02 *= pixels[pixelIndex(x-1,y+1,width)];
    x20 *= pixels[pixelIndex(x+1,y-1,width)];
    x21 *= pixels[pixelIndex(x+1,y,width)];
    x22 *= pixels[pixelIndex(x+1,y+1,width)];
    
    int y00 = -1;  int y10 = -2;  int y20 = -1;
    int y02 = 1;  int y12 = 2;  int y22 = 1;
    y00 *= pixels[pixelIndex(x-1,y-1,width)];
    y10 *= pixels[pixelIndex(x,y-1,width)];
    y20 *= pixels[pixelIndex(x+1,y-1,width)];
    y02 *= pixels[pixelIndex(x-1,y+1,width)];
    y12 *= pixels[pixelIndex(x,y+1,width)];
    y22 *= pixels[pixelIndex(x+1,y+1,width)];
    int pixelX = x00 + x01 + x02 + x20 + x21 + x22;
    int pixelY = y00 + y10 + y20 + y02 + y12 + y22;
    arr[pixelIndex(x,y,width)] = (int)sqrtf(pixelX*pixelX +pixelY*pixelY);
  }else{
    arr[pixelIndex(x,y,width)] = pixels[pixelIndex(x,y,width)];
  }
}

int main()
{
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);

  // Load image and get the width and height
  FIBITMAP *image;
  image = FreeImage_Load(FIF_PNG, "coins.png", 0);
  if (image == NULL)
  {
      printf("Image Load Problem\n");
      exit(0);
  }
  int imgWidth;
  int imgHeight;
  imgWidth = FreeImage_GetWidth(image);
  imgHeight = FreeImage_GetHeight(image);

  // Convert image into a flat array of chars with the value 0-255 of the
  // greyscale intensity
  int *arr;
  int *dev_arr;
  arr = (int *)malloc(sizeof(int) * imgWidth *imgHeight);
  cudaMalloc((void **)&dev_arr, sizeof(int) * imgWidth * imgHeight);

  RGBQUAD aPixel;
  char *pixels;  
  char *dev_pixels;
  int pixIndex = 0;
  pixels = (char *) malloc(sizeof(char)*imgWidth*imgHeight);
  for (int i = 0; i < imgHeight; i++){
    for (int j = 0; j < imgWidth; j++)
    {
      FreeImage_GetPixelColor(image,j,i,&aPixel);
      char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
      pixels[pixIndex++]=grey;
    }
  }
  cudaMalloc((void **)&dev_pixels, sizeof(char) * imgWidth *imgHeight);
  cudaMemcpy(dev_pixels, pixels, sizeof(char) * imgWidth *imgHeight, cudaMemcpyHostToDevice);
  int powOfTwo = 2;
  while ((imgHeight % powOfTwo) == 0 && (imgWidth % powOfTwo) == 0){
    powOfTwo *= 2;
  }
  dim3 numThreadsPerBlock(powOfTwo,powOfTwo);
  int blockX = imgHeight / powOfTwo;
  int blockY = imgWidth / powOfTwo;
  dim3 numBlocks(blockX,blockY);
  // Apply sobel operator to pixels, ignoring the borders
  sobel<<<numBlocks, numThreadsPerBlock>>>(imgWidth, imgHeight, dev_pixels, dev_arr);
  cudaMemcpy(arr,dev_arr,sizeof(int) * imgWidth * imgHeight, cudaMemcpyDeviceToHost);

  FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
  for (int i = 1; i < imgWidth-1; i++)
  {
    for (int j = 1; j < imgHeight-1; j++)
    {
    int sVal = arr[j * imgWidth + i];
      aPixel.rgbRed = sVal;
      aPixel.rgbGreen = sVal;
      aPixel.rgbBlue = sVal;
      FreeImage_SetPixelColor(bitmap, i, j, &aPixel);
    }
  }
  FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);
  free(arr);
  cudaFree(dev_arr);
  free(pixels);
  cudaFree(dev_pixels);
  FreeImage_Unload(bitmap);
  FreeImage_Unload(image);
  return 0;
}
