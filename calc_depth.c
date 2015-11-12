/*
 * PROJ1-1: YOUR TASK A CODE HERE
 *
 * Feel free to define additional helper functions.
 */

#include "calc_depth.h"
#include "utils.h"
#include <math.h>
#include <limits.h>
#include <stdio.h>

#include <stdlib.h>

/* Implements the normalized displacement function */
unsigned char normalized_displacement(int dx, int dy,
        int maximum_displacement) {

    double squared_displacement = dx * dx + dy * dy;
    double normalized_displacement = round(255 * sqrt(squared_displacement) / sqrt(2 * maximum_displacement * maximum_displacement));
    return (unsigned char) normalized_displacement;

}

unsigned char mat_val(unsigned char *img, int image_width, int image_height, int x, int y){
  return img[y * image_width + x];
}

void set_val(unsigned char *img, int image_width, int image_height, int x, int y, unsigned char val){
  img[y * image_width + x] = val;
  //*(img + y * image_width * sizeof(unsigned char) + x * sizeof(unsigned char)) = val;
}

int inside_img(int image_width, int image_height, int feature_width, int feature_height, int x, int y){
  if(x < 0 || y < 0){
    return 0;
  }

  if(feature_height > y || feature_height > (image_height - 1 - y)){
    return 0;
  }

  if(feature_width > x || feature_width > (image_width - 1 - x)){
    return 0;
  }

  return 1;
}

int pixel_inside_img(int image_width, int image_height, int x, int y){
  return x >= 0 && x < image_width && y >= 0 && y < image_height;
}

void calc_depth(unsigned char *depth_map, unsigned char *left,
        unsigned char *right, int image_width, int image_height,
        int feature_width, int feature_height, int maximum_displacement) {

  //for every pixel
  for(int row = 0; row < image_height; row++){
    for(int col = 0; col < image_width; col++){
      //scan larger area in right image
      int center_x = 0;
      int center_y = 0;
      unsigned int lowest_dist = ~0;
      unsigned int lowest_sum = ~0;

      if(!inside_img(image_width, image_height, feature_width, feature_height, col, row)){
        set_val(depth_map, image_width, image_height, col, row, 0);
      } else{
        for (int i = -maximum_displacement; i < maximum_displacement + 1; i++) {
          for (int j = -maximum_displacement; j < maximum_displacement + 1; j++) {
            int search_x = col + j;
            int search_y = row + i;

            if(inside_img(image_width, image_height, feature_width, feature_height, search_x, search_y)){

              //calculate sum
              unsigned int sum = 0;
              int in_bounds = 1;
              for (int k = -feature_height; k < feature_height + 1; k++) {
                for (int l = -feature_width; l < feature_width + 1; l++) {
                  if(pixel_inside_img(image_width, image_height, search_x + l, search_y + k)){
                    unsigned char right_val = mat_val(right, image_width, image_height, search_x + l, search_y + k);
                    unsigned char left_val = mat_val(left,  image_width,  image_height, col + l, row + k);
                    int difference = (right_val > left_val) ? right_val - left_val : left_val - right_val;
                    sum += difference * difference;
                  } else{
                    in_bounds = 0;
                    break;
                  }
                }

                if(!in_bounds){
                  break;
                }
              }

              //update lowest sum after sum is calculated
              if(sum <= lowest_sum && in_bounds){
                int temp_dist = normalized_displacement(col - search_x, row - search_y, maximum_displacement);
                if(sum == lowest_sum){
                  if(temp_dist < lowest_dist){
                    center_x = search_x;
                    center_y = search_y;
                    lowest_dist = temp_dist;
                  }
                } else{
                  center_x = search_x;
                  center_y = search_y;
                  lowest_dist = temp_dist;
                  lowest_sum = sum;
                }
              }
            }
          }
        }

        int dy = row - center_y;
        int dx = col - center_x;
        unsigned char norm_dp = normalized_displacement(dx, dy, maximum_displacement);
        set_val(depth_map, image_width, image_height, col, row, norm_dp);
      }
    }
  }
}


