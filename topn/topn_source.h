#ifndef TOPN_SOURCE_CPPCLASS_H
#define TOPN_SOURCE_CPPCLASS_H

template<typename T>
struct Candidate {
	int index;
	T value;

	bool operator<(const Candidate& a) const
    {
        return a.value < value;
    }

};

template<typename T>
void sparse_topn_source(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int ntop,
		int Bp[],
		int Bj[],
		T Bx[]	//data of output
);

#endif //TOPN_SOURCE_CPPCLASS_H
