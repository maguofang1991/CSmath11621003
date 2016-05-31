#include "c_eval.h"
#include <iostream>
using namespace std;
c_eval::c_eval(){
    num_factors = 0; // m_num_topics
    num_items = 0; // m_num_docs
    num_users = 0; // num of users
    best_mae = 9999;best_rsme = 9999;

}

c_eval::~c_eval() {
  // free memory
 
}
void c_eval::set_parameters(char* directory,int num_factors,int num_users,int num_items){
	this->num_factors = num_factors;
        this->num_users = num_users;
        this->num_items = num_items;

	// initializing the saving file
	strcpy(mae_path,directory);
	strcat(mae_path,"/mae.txt");
	strcpy(rsme_path,directory);
	strcat(rsme_path,"/rsme.txt");
}







// calculate the mae and rsme
double c_eval::cal_mae( ){
	FILE *resultFile;

	resultFile = fopen(mae_path,"w");
	fprintf(resultFile,"\nThe mae is %lf.\n",best_mae);
	fclose(resultFile);
	return best_mae;
}
double c_eval::cal_rsme(){
	FILE *resultFile;

	resultFile = fopen(rsme_path,"w");
	fprintf(resultFile,"\nThe rsme is %lf.\n",best_rsme);
	fclose(resultFile);
	return best_rsme;
}


double c_eval::mtx_mae(gsl_matrix* m_U,gsl_matrix* m_V, const c_data* test_users,int num_test_users)
{
	double mae = 0.0,sum = 0.0,tRate = 0.0,iRate = 0.0;
	int i, j,l,n,s,cnt = 0;
	int* item_ids; 
  	double* item_scores;
	
	for (i = 0; i < num_test_users; i ++) {
		gsl_vector_view u = gsl_matrix_row(m_U, i);
		item_ids = test_users->m_vec_data[i];
     		item_scores = test_users->m_vec_score[i];

    	  	n = test_users->m_vec_len[i];
		if (n > 0) { // this user has rated some articles
			for (l=0; l < n; l ++) {
			  j = item_ids[l];
			  s = item_scores[l];
			  tRate = s/1.0;
			  gsl_vector_view v = gsl_matrix_row(m_V, j);
			  gsl_blas_ddot(&u.vector, &v.vector, &iRate);
			  sum +=  fabs(tRate - iRate);cnt++;
			  
			}
		}

	}

	mae = sum/cnt;
	if (mae<best_mae) best_mae=mae;
	return mae;
}

double c_eval::mtx_rsme(gsl_matrix* m_U,gsl_matrix* m_V, const c_data* test_users,int num_test_users)
{
	double rsme = 0.0,sum = 0.0,tRate = 0.0,iRate = 0.0;
	int i, j,l,n,s,cnt = 0;
	int* item_ids; 
  	double* item_scores;
	for (i = 0; i < num_test_users; i ++) {
		gsl_vector_view u = gsl_matrix_row(m_U, i);
		item_ids = test_users->m_vec_data[i];
     		item_scores = test_users->m_vec_score[i];

    	  	n = test_users->m_vec_len[i];
		if (n > 0) { // this user has rated some articles
			for (l=0; l < n; l ++) {
			  j = item_ids[l];
			  s = item_scores[l];
			  tRate = s/1.0;
			  gsl_vector_view v = gsl_matrix_row(m_V, j);
			  gsl_blas_ddot(&u.vector, &v.vector, &iRate);
			  sum = sum + (tRate - iRate)*(tRate - iRate);
			  cnt++;
			}
		}

	}

	rsme = sqrt(sum/cnt);if (rsme<best_rsme) best_rsme=rsme;
	return rsme;
}


