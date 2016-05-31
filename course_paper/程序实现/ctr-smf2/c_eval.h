// class for eval

#ifndef C_EVAL_H
#define C_EVAL_H

#include "utils.h"
#include "data.h"

class c_eval
{
public:
	c_eval();
	~c_eval();
	void set_parameters(char* directory,int num_factors,int num_users,int num_items);
	double mtx_mae(gsl_matrix* m_U,gsl_matrix* m_V, const c_data* test_users,int num_test_users);
	double mtx_rsme(gsl_matrix* m_U, gsl_matrix* m_V, const c_data* test_users,int num_test_users);
	double cal_mae();
	double cal_rsme();

public:
	int num_factors;
	int num_users;
	int num_items;
	double best_mae;
	double best_rsme;
	char mae_path[100];
	char rsme_path[100];

};

#endif
