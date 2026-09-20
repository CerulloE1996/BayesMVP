

#pragma once




 

inline  std::vector< Eigen::Matrix<double, -1, -1>> vec_of_mats_double(int n_rows,
                                                                      int n_cols, 
                                                                      int n_matrices) {
  
        std::vector< Eigen::Matrix<double, -1, -1> > vec_of_mats(n_matrices);
        const Eigen::Matrix<double, -1, -1> &zero_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
        
        for (int i = 0; i < n_matrices; ++i) {
          vec_of_mats[i] = zero_mat;
        }  
         
        return vec_of_mats;
  
}




template<typename T = double>
inline std::vector<Eigen::Matrix<T, -1, -1>> vec_of_mats(int n_rows,
                                                         int n_cols, 
                                                         int n_mats) {
  
        std::vector<Eigen::Matrix<T, -1, -1>> result;
        result.reserve(n_mats);
        
        const Eigen::Matrix<T, -1, -1> &zero_mat = Eigen::Matrix<T, -1, -1>::Zero(n_rows, n_cols);
        
        for (int i = 0; i < n_mats; ++i) {
          result.emplace_back(zero_mat);
        } 
        
        return result;
  
}




template<typename T = double>
inline std::vector<std::vector<Eigen::Matrix<T, -1, -1>>> vec_of_vec_of_mats(int n_rows,
                                                                             int n_cols, 
                                                                             int n_mats_inner,
                                                                             int n_mats_outer
) {
  
        std::vector<std::vector<Eigen::Matrix<T, -1, -1>>> result;
        result.reserve(n_mats_outer);
        
        const Eigen::Matrix<T, -1, -1> &zero_mat = Eigen::Matrix<T, -1, -1>::Zero(n_rows, n_cols);
        
        for (int i = 0; i < n_mats_outer; ++i) {
            
            std::vector<Eigen::Matrix<T, -1, -1>> inner_vec;
            inner_vec.reserve(n_mats_inner);
            
            for (int j = 0; j < n_mats_inner; ++j) {
              inner_vec.emplace_back(zero_mat);
            }
            result.emplace_back(std::move(inner_vec)); 
          
        }
        
        return result;
  
}




// input vector, outputs upper-triangular 3d array of corrs- double
inline std::vector<Eigen::Matrix<double, -1, -1 > >   fn_convert_std_vec_of_corrs_to_3d_array_double(
     const std::vector<double>   &input_vec,
     int n_rows,
     int n_arrays
) {
   
         std::vector<Eigen::Matrix<double, -1, -1 > >   output_array = vec_of_mats(n_rows, n_rows, n_arrays); // 1d vector to output
         
         int k = 0;
         for (int c = 0; c < n_arrays; ++c) {
           for (int i = 1; i < n_rows; ++i)  {
             for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
               output_array[c](i, j) =  input_vec[k];
               k += 1;
             }
           } 
         }
         
         return output_array; // output is a parameter to use in the log-posterior function to be differentiated
   
}




// input vector, outputs upper-triangular 3d array of corrs- double
inline std::vector<Eigen::Matrix<double, -1, -1 > >   fn_convert_Eigen_vec_of_corrs_to_3d_array_double(
    const Eigen::Matrix<double, -1, 1 >   &input_vec,
    int n_rows,
    int n_arrays
) {
  
        std::vector<Eigen::Matrix<double, -1, -1 > >   output_array = vec_of_mats(n_rows, n_rows, n_arrays); // 1d vector to output
        
        int k = 0;
        for (int c = 0; c < n_arrays; ++c) {
          for (int i = 1; i < n_rows; ++i)  {
            for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R 
              output_array[c](i, j) =  input_vec(i);
              k += 1;
            }
          }
        } 
        
        return output_array; // output is a parameter to use in the log-posterior function to be differentiated
  
}










 
 