#include <omp.h>
#include <iostream>
#include <math.h>
#include <locale.h>
#include <ctime>


using namespace std;

int main() {
	int i, num_threads = 1;
	setlocale(LC_ALL, "Rus");

	double eps = 1e-15; // -28, -26, -24, -22

	double x_prev50[5][50], x50[5][50] = {0}, b50[5][50], b_50[5][50] = {0};
	double x_prev100[5][100], x100[5][100] = {0}, b100[5][100], b_100[5][100] = {0};
	double x_prev500[5][500], x500[5][500] = {0}, b500[5][500], b_500[5][500] = {0};
	double x_prev2000[5][2000], x2000[5][2000] = {0}, b2000[5][2000], b_2000[5][2000] = {0};

	for (int j = 0; j < 5; ++j) {
		for (i = 0; i < 2000; ++i) {
			if (i < 50) {
				x_prev50[j][i] = 1; x_prev100[j][i] = 1; x_prev500[j][i] = 1; x_prev2000[j][i] = 1;
				b50[j][i] = i; b100[j][i] = i; b500[j][i] = i; b2000[j][i] = i;
			}
			else if (i < 100) {
				x_prev100[j][i] = 1; x_prev500[j][i] = 1; x_prev2000[j][i] = 1;
				b100[j][i] = i; b500[j][i] = i; b2000[j][i] = i;
			}
			else if (i < 500) {
				x_prev500[j][i] = 1; x_prev2000[j][i] = 1;
				b500[j][i] = i; b2000[j][i] = i;
			}
			else {
				x_prev2000[j][i] = 1;
				b2000[j][i] = i;
			}
		}
	}
		
	double norm50 = 1, norm100 = 1, norm500 = 1, norm2000 = 1, res[4][5], cur;
	double dt[4][5];
	clock_t t1 = clock();
	clock_t t2 = clock();

	// 50
	for (i = 0; i < 5; ++i) {
		t1 = clock();

		norm50 = 1;
		while (norm50 >= eps) {
			norm50 = 0;
			num_threads = pow(2, i);
			#pragma omp parallel num_threads(num_threads) reduction(+:norm50)
			{
				int id = omp_get_thread_num();
				int size = omp_get_num_threads();
				int step = 50 / size;
				int start = id * step;
				int end = (id + 1) * step;
				if (id == size - 1) end = 50;

				double s1 = 0, s2 = 0;

				for (int j = 0; j < 50; ++j) {
					s1 = 0;
					s2 = 0;
					for (int k = 0; k < j; ++k) s1 += 1 * x_prev50[i][k];
					for (int k = j + 1; k < 50; ++k) s2 += 1 * x_prev50[i][k];
					x50[i][j] = (b50[i][j] - s1 - s2) / 50;
				}
				for (int j = start; j < end; ++j) {
					norm50 += (x50[i][j] - x_prev50[i][j]) * (x50[i][j] - x_prev50[i][j]);
				}
			}
			std::swap(x_prev50[i], x50[i]);
		}
		t2 = clock();
		dt[0][i] = (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC) * 1000000;

		for (int j = 0; j < 50; ++j) {
			for (int k = 0; k < 50; ++k) {
				if (j == k) b_50[i][j] += x50[i][k] * 50;
				else b_50[i][j] += x50[i][k] * 1;
			}
		}

		res[0][i] = 0;
		for (int j = 0; j < 50; ++j) {
			cur = fabs(b50[i][j] - b_50[i][j]);
			res[0][i] = (res[0][i] <= cur) ? cur : res[0][i];
		}
	}
	
	for (int j = 0; j < 5; ++j) {
		cout << "Невязка: " << res[0][j] << ", " << "время: " << dt[0][j] << endl;
	}
	cout << endl;



	// 100
	for (i = 0; i < 5; ++i) {
		t1 = clock();

		norm100 = 1;
		while (norm100 >= eps) {
			norm100 = 0;
			num_threads = pow(2, i);
			#pragma omp parallel num_threads(num_threads) reduction(+:norm100)
			{
				int id = omp_get_thread_num();
				int size = omp_get_num_threads();
				int step = 100 / size;
				int start = id * step;
				int end = (id + 1) * step;
				if (id == size - 1) end = 100;

				double s1 = 0, s2 = 0;

				for (int j = start; j < end; ++j) {
					s1 = 0;
					s2 = 0;
					for (int k = 0; k < j; ++k) s1 += 1 * x_prev100[i][k];
					for (int k = j + 1; k < 100; ++k) s2 += 1 * x_prev100[i][k];
					x100[i][j] = (b100[i][j] - s1 - s2) / 100;
				}

				for (int j = start; j < end; ++j) {
					norm100 += (x100[i][j] - x_prev100[i][j]) * (x100[i][j] - x_prev100[i][j]);
				}
			}

			std::swap(x_prev100[i], x100[i]);
		}
		t2 = clock();
		dt[1][i] = (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC) * 1000000;

		for (int j = 0; j < 100; ++j) {
			for (int k = 0; k < 100; ++k) {
				if (j == k) b_100[i][j] += x100[i][k] * 100;
				else b_100[i][j] += x100[i][k] * 1;
			}
		}

		res[1][i] = 0;
		for (int j = 0; j < 100; ++j) {
			cur = fabs(b100[i][j] - b_100[i][j]);
			res[1][i] = (res[1][i] <= cur) ? cur : res[1][i];
		}
	}

	for (int j = 0; j < 5; ++j) {
		cout << "Невязка: " << res[1][j] << ", " << "время: " << dt[1][j] << endl;
	}
	cout << endl;

	// 500
	for (i = 0; i < 5; ++i) {
		t1 = clock();

		norm500 = 1;
		while (norm500 >= eps) {
			norm500 = 0;
			num_threads = pow(2, i);
			#pragma omp parallel num_threads(num_threads) reduction(+:norm500)
			{
				int id = omp_get_thread_num();
				int size = omp_get_num_threads();
				int step = 500 / size;
				int start = id * step;
				int end = (id + 1) * step;
				if (id == size - 1) end = 500;

				double s1 = 0, s2 = 0;

				for (int j = start; j < end; ++j) {
					s1 = 0;
					s2 = 0;
					for (int k = 0; k < j; ++k) s1 += 1 * x_prev500[i][k];
					for (int k = j + 1; k < 500; ++k) s2 += 1 * x_prev500[i][k];
					x500[i][j] = (b500[i][j] - s1 - s2) / 500;
				}

				for (int j = start; j < end; ++j) {
					norm500 += (x500[i][j] - x_prev500[i][j]) * (x500[i][j] - x_prev500[i][j]);
				}
			}

			std::swap(x_prev500[i], x500[i]);
		}
		t2 = clock();
		dt[2][i] = (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC) * 1000000;

		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 500; ++k) {
				if (j == k) b_500[i][j] += x500[i][k] * 500;
				else b_500[i][j] += x500[i][k] * 1;
			}
		}

		res[2][i] = 0;
		for (int j = 0; j < 500; ++j) {
			cur = fabs(b500[i][j] - b_500[i][j]);
			res[2][i] = (res[2][i] <= cur) ? cur : res[2][i];
		}
	}

	for (int j = 0; j < 5; ++j) {
		cout << "Невязка: " << res[2][j] << ", " << "время: " << dt[2][j] << endl;
	}
	cout << endl;

	// 2000
	for (i = 0; i < 5; ++i) {
		t1 = clock();

		norm2000 = 1;
		while (norm2000 >= eps) {
			norm2000 = 0;
			num_threads = pow(2, i);
			#pragma omp parallel num_threads(num_threads) reduction(+:norm2000)
			{
				int id = omp_get_thread_num();
				int size = omp_get_num_threads();
				int step = 2000 / size;
				int start = id * step;
				int end = (id + 1) * step;
				if (id == size - 1) end = 2000;

				double s1 = 0, s2 = 0;

				for (int j = start; j < end; ++j) {
					s1 = 0;
					s2 = 0;
					for (int k = 0; k < j; ++k) s1 += 1 * x_prev2000[i][k];
					for (int k = j + 1; k < 2000; ++k) s2 += 1 * x_prev2000[i][k];
					x2000[i][j] = (b2000[i][j] - s1 - s2) / 2000;
				}

				for (int j = start; j < end; ++j) {
					norm2000 += (x2000[i][j] - x_prev2000[i][j]) * (x2000[i][j] - x_prev2000[i][j]);
				}
			}

			std::swap(x_prev2000[i], x2000[i]);
		}
		t2 = clock();
		dt[3][i] = (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC) * 1000000;

		for (int j = 0; j < 2000; ++j) {
			for (int k = 0; k < 2000; ++k) {
				if (j == k) b_2000[i][j] += x2000[i][k] * 2000;
				else b_2000[i][j] += x2000[i][k] * 1;
			}
		}

		res[3][i] = 0;
		for (int j = 0; j < 2000; ++j) {
			cur = fabs(b2000[i][j] - b_2000[i][j]);
			res[3][i] = (res[3][i] <= cur) ? cur : res[3][i];
		}
	}
	
	for (int j = 0; j < 5; ++j) {
		cout << "Невязка: " << res[3][j] << ", " << "время: " << dt[3][j] << endl;
	}

	return 0;
} 
