

#pragma once




 

ALWAYS_INLINE  std::vector< Eigen::Matrix<double, -1, -1>> vec_of_mats_double(int n_rows,
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
ALWAYS_INLINE std::vector<Eigen::Matrix<T, -1, -1>> vec_of_mats(int n_rows,
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
ALWAYS_INLINE std::vector<std::vector<Eigen::Matrix<T, -1, -1>>> vec_of_vec_of_mats(int n_rows,
                                                                             int n_cols, 
                                                                             int n_mats_inner,
                                                                             int n_mats_outer) {
  
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



 
 





 
 