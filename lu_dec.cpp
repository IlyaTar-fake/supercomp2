#include <iostream>
#include <math.h>

using namespace std;

// Ѕлоки 5х5;  n = 100, r = 5

#define n 10
#define r 5

static double A[n][n], U[n][n], L[n][n];

void crout_lu(int i) {
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

void right_up(int i, int j) {
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
void left_down(int i, int j) {
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
void red(int i) {
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

void lu(int i) {
	if (i <= r) 
		crout_lu(i);

#pragma omp task priority(2)
	crout_lu(i);
#pragma omp task priority(1)
	right_up(i + r, i);
#pragma omp task priority(1)
	left_down(i, i + r);
#pragma omp taskwait

	red(i + r);
	lu(i + r);
}

int main() {
#pragma omp for
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i][j] = (i == j) ? n : 1;
			U[i][j] = 0;
			L[i][j] = (i == j) ? 1 : 0;
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 && j == 0) cout << "A: ";
			printf("%.0lf", A[i][j]);
			if ((i != n - 1) || (j != n - 1)) cout << " ";
			else cout << endl;
		}
		cout << "\n   ";
	}
	cout << endl;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 && j == 0) cout << "L: ";
			printf("%.3lf", L[i][j]);
			if ((i != n - 1) || (j != n - 1)) cout << " ";
			else cout << endl;
		}
		cout << "\n   ";
	}
	cout << endl;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 && j == 0) cout << "U: ";
			printf("%.3lf", U[i][j]);
			if ((i != n - 1) || (j != n - 1)) cout << " ";
			else cout << endl;
		}
		cout << "\n   ";
	}
	cout << endl;


#pragma omp parallel
#pragma omp single nowait
	lu(0);


	double s1 = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			s1 = 0;
			for (int s = 0; s < n; ++s)
				s1 += L[i][s] * U[s][j];
			A[i][j] = s1;
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == 0 && j == 0) cout << "A: ";
			printf("%.0lf", A[i][j]);
			if ((i != n - 1) || (j != n - 1)) cout << " ";
			else cout << endl;
		}
		cout << "\n   ";
	}
	cout << endl;

	return 0;
}
