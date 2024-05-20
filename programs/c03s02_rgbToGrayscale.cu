#include <stdio.h>
#include <jpeglib.h>
#include <cuda_runtime.h>

__host__
void loadImage(const char* filename, unsigned char** image, int* width, int* height) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* infile;
    JSAMPARRAY buffer;

    // Open file
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        exit(1);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *image = (unsigned char*)malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, cinfo.output_width * cinfo.output_components, 1);

    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(*image + (cinfo.output_scanline - 1) * cinfo.output_width * cinfo.output_components, buffer[0], cinfo.output_width * cinfo.output_components);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}

__host__
void saveGrayJPEG(const char* filename, unsigned char* grayData, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile;
    JSAMPROW row_pointer[1];

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Can't open %s for writing\n", filename);
        exit(1);
    }

    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &grayData[cinfo.next_scanline * width];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}


__global__
void convertRGBPixelToGrayscale(
    unsigned char *gray,
    unsigned char *rgb,
    int width, 
    int height
) {
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < height) {
        int i = row * width + col;
        unsigned char r = rgb[3 * i];
        unsigned char g = rgb[3 * i + 1];
        unsigned char b = rgb[3 * i + 2];
        gray[i] = r*0.21f + g*0.72f + b*0.07f;
    }
}


int main() {
    int width, height;
    unsigned char *rgb_h, *rgb_d, *gray_d;

    loadImage("elf_rgb.jpeg", &rgb_h, &width, &height);

    cudaMalloc(&rgb_d, width * height * 3);
    cudaMalloc(&gray_d, width * height);
    cudaMemcpy(rgb_d, rgb_h, width * height * 3, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + 15) / 16, (height + 15) / 16);
    convertRGBPixelToGrayscale<<<dimGrid, dimBlock>>>(gray_d, rgb_d, width, height);

    unsigned char* gray_h = (unsigned char*)malloc(width * height);
    cudaMemcpy(gray_h, gray_d, width * height, cudaMemcpyDeviceToHost);

    // Save to JPEG
    saveGrayJPEG("elf_gray.jpeg", gray_h, width, height);

    cudaFree(rgb_d);
    cudaFree(gray_d);
    free(rgb_h);
    free(gray_h);

    return 0;
}