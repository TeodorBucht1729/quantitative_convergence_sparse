#include <vector>
#include <assert.h>
#include<iostream>
#include<numeric>
#include <gmp.h>

using namespace std;

#define rep(i, a, b) for(int i = a; i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

typedef vector<double> vd;

int solveLinear(vector<vector<mpq_t>>& A, vector<mpq_t>& b, vector<mpq_t>& x) {
	int n = sz(A), m = sz(x), rank = 0, br, bc;
	if (n) assert(sz(A[0]) == m);
	vi col(m); iota(all(col), 0);
	mpq_t zero_rational;
	mpq_t fac, to_sub;
	mpq_init(fac);
	mpq_init(to_sub);
	mpq_init(zero_rational);
	bool found_non_zero = false;
	int percentage_done;
	percentage_done = 0;
	rep(i,0,n) {
		int new_percentage;
		new_percentage = (i * 100) / n;
		// if (new_percentage > percentage_done) {
		// 	cout << new_percentage << "% done" << endl;
		// 	percentage_done = new_percentage;
		// }
		mpq_t v, bv;
		mpq_init(v);
		mpq_init(bv);
		found_non_zero = false;
		rep(r,i,n) {
			rep(c,i,m) {
				// mpq_set(v, A[r][c]);
				if (mpq_equal(A[r][c], zero_rational) == 0){
					br = r, bc = c, found_non_zero = true;
					break;
				}
			}
			if (found_non_zero) break;
		}
		if (!found_non_zero) {
			rep(j,i,n) if (mpq_equal(b[j], zero_rational) == 0) return -1;
			break;
		}
		swap(A[i], A[br]);
		swap(b[i], b[br]);
		swap(col[i], col[bc]);
		rep(j,0,n) swap(A[j][i], A[j][bc]);
		// cout << "Dividing by: ";
		// mpq_out_str(stdout, 10, A[i][i]); 
		// cout << endl;
		mpq_inv(bv, A[i][i]);

		rep(j,i+1,n) {
			mpq_mul(fac, A[j][i], bv); 
			if (mpq_equal(fac, zero_rational) != 0){
				continue;
			}
			mpq_mul(to_sub, fac, b[i]);
			mpq_sub(b[j], b[j], to_sub);
			// b[j] -= fac * b[i];
			rep(k,i+1,m){
				mpq_mul(to_sub, fac, A[i][k]);
				mpq_sub(A[j][k], A[j][k], to_sub);
				// A[j][k] -= fac*A[i][k];
			} 
		}
		rank++;
	}

	rep(i, 0, m) {
		mpq_init(x[i]);
	}
	for (int i = rank; i--;) {
		mpq_div(b[i], b[i], A[i][i]);
		mpq_set(x[col[i]], b[i]);
		rep(j, 0, i) {
			mpq_mul(to_sub, A[j][i], b[i]);
			mpq_sub(b[j], b[j], to_sub);
		}
		// b[i] /= A[i][i];
		// x[col[i]] = b[i];
		// rep(j,0,i) b[j] -= A[j][i] * b[i];
	}
	return rank; // (multiple solutions if rank < m)
}

int main() {
	// cin.tie(0)->sync_with_stdio(0);
	// cin.exceptions(cin.failbit);
	ll n, m, p, nbr_non_zero;
	cin >> m >> n >> p >> nbr_non_zero;
	vector<vector<mpq_t>> A (m);

	vector<mpq_t> b(m);
	vector<mpq_t> x(n);
	rep(i, 0, m) {
		A[i] = vector<mpq_t> (n);
		rep(j, 0, n) {
			mpq_init(A[i][j]);
		}
		mpq_init(b[i]);
	}
	rep(ind, 0, nbr_non_zero) {
		ll i, j;
		int p;
		unsigned int q;
		cin >> i >> j >> p >> q;
		if (j > n) {
			mpq_set_si(b[i-1], p, q);
		} else {
			mpq_set_si(A[i-1][j-1], p, q);
		}
		
	}
	int rank;
	rank = solveLinear(A, b, x);
	if (rank == -1) {
		cout << "No solution!" << endl;
		return 0;
	}
	// cout << "rank: " << rank << endl;
	rep(i, 0, n) {
		mpq_out_str(stdout, 10, x[i]);
		if (i < n-1) cout << endl;
	}
	
}

