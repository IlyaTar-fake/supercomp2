#include <iostream>
#include <math.h>
#include <stdio.h>
#include <ctime>

using namespace std;

const int n = 5000;
const int r = 5;

static double A[n][n], U[n][n], L[n][n];

static void crout_lu(int i) {
	int end = (i + r < n) ? i + r : n;

	double s1 = 0;
	for (int l = i; l < end; ++l) {
		for (int k = i; k <= l; ++k) {
			s1 = 0;
			for (int s = i; s < k; ++s)
				s1 += L[k][s] * U[s][l];
			U[k][l] = A[k][l] - s1;
		}
		for (int k = i + 1; k < end; ++k) {
			s1 = 0;
			for (int s = i; s < l; ++s)
				s1 += L[k][s] * U[s][l];
			L[k][l] = (A[k][l] - s1) / U[l][l];
		}
	}
}
static void right_up(int i, int j) {
	double s1 = 0;
	for (int l = j; l < n; ++l) {
		for (int k = i; k < i + r; ++k) {
			s1 = 0;
			for (int s = i; s < k; ++s)
				s1 += L[k][s] * U[s][l];
			U[k][l] = A[k][l] - s1;
		}
	}
}
static void left_down(int i, int j) {
	double s1 = 0;
	for (int k = i; k < n; ++k) {
		for (int l = j + r - 1; l >= j; --l) {
			s1 = 0;
			for (int s = l + 1; s < j + r; ++s)
				s1 += L[k][s] * U[s][l];
			L[k][l] = (A[k][l] - s1) / U[l][l];
		}
	}
}
static void red(int i) {
	double s1 = 0;
	for (int k = i; k < n; ++k) {
		for (int l = i; l < n; ++l) {
			s1 = 0;
			for (int s = i - r; s < i; ++s)
				s1 += L[k][s] * U[s][l];
			A[k][l] -= s1;
		}
	}
}

static void lu(int i) {
	if (n - i <= r)
		crout_lu(i);
	if (i >= n) return;

#pragma omp task
	crout_lu(i);
#pragma omp taskwait
#pragma omp task
	right_up(i, i + r);
#pragma omp task
	left_down(i + r, i);
#pragma omp taskwait

	red(i + r);
	lu(i + r);
}

int main() {
	for (int num = 0; num < 5; ++num) {

#pragma omp for
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				A[i][j] = (i == j) ? n : 1;
				U[i][j] = 0;
				L[i][j] = (i == j) ? 1 : 0;
			}
		}

		int num_threads = pow(2, num);
		clock_t t1 = clock();

#pragma omp parallel num_threads(num_threads)
#pragma omp single nowait
		lu(0);

		clock_t t2 = clock();

		double s1 = 0;
		double res = 0, res_curr;

#pragma omp for private(res_curr, s1)
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				s1 = 0;
				for (int s = 0; s < n; ++s)
					s1 += L[i][s] * U[s][j];
				res_curr = fabs(A[i][j] - s1);
#pragma omp critical 
				{
					res = (res > res_curr) ? res : res_curr;
				}
			}
		}

		cout << "Residual: " << res << "; ";
		cout << "time: ";
		printf("%.6lf\n", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
	}

	return 0;
}