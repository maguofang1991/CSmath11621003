#include <assert.h>
#include <iostream>
#include <stdio.h>
#include "data.h"

using namespace std;

/**
 * rating matrix
 * each record is a person, and each array contains items.
 */
c_data::c_data() {
}

c_data::~c_data() {
  for (size_t i = 0; i < m_vec_data.size(); i ++) {
    int* ids = m_vec_data[i];
    if (ids != NULL) delete [] ids;
    double* scores = m_vec_score[i];
    if(scores != NULL) delete[] scores;
  }
  m_vec_data.clear();
  m_vec_len.clear();
  m_vec_score.clear();
}

void c_data::read_data(const char * data_filename, int OFFSET) {

  int length = 0, n = 0, id = 0, total = 0;
  double score = 0;

  FILE * fileptr;
  fileptr = fopen(data_filename, "r");

  while ((fscanf(fileptr, "%10d", &length) != -1)) {
    int * ids = NULL;
    double * scores = NULL;
    if (length > 0) {
      ids = new int[length];
      scores = new double[length];
      for (n = 0; n < length; n++) {
        fscanf(fileptr, "%10d:%10lf", &id, &score);
        ids[n] = id - OFFSET;
        scores[n] = score;
      }
    }
    m_vec_data.push_back(ids);
    m_vec_len.push_back(length);
    m_vec_score.push_back(scores);
    total += length;
  }
  fclose(fileptr);
  printf("read %d vectors with %d entries ...\n", (int)m_vec_len.size(), total);
}

