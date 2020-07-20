#ifndef scalapack_templates_h
#define scalapack_templates_h



#include <deal.II/base/config.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/exceptions.h>



extern "C"
{

  void
  Cblacs_pinfo(int *rank, int *nprocs);

  void
  Cblacs_get(int icontxt, int what, int *val);

  void
  Cblacs_gridinit(int *       context,
                  const char *order,
                  int         grid_height,
                  int         grid_width);

  void
  Cblacs_gridinfo(int  context,
                  int *grid_height,
                  int *grid_width,
                  int *grid_row,
                  int *grid_col);

  void
  Cblacs_pcoord(int ictxt, int pnum, int *prow, int *pcol);

  void
  Cblacs_gridexit(int context);

  void
  Cblacs_barrier(int, const char *);

  void
  Cblacs_exit(int error_code);

  void
  Cdgerv2d(int context, int M, int N, double *A, int lda, int rsrc, int csrc);

  void
  Csgerv2d(int context, int M, int N, float *A, int lda, int rsrc, int csrc);

  void
  Cdgesd2d(int context, int M, int N, double *A, int lda, int rdest, int cdest);

  void
  Csgesd2d(int context, int M, int N, float *A, int lda, int rdest, int cdest);

  int
  Csys2blacs_handle(MPI_Comm comm);

  int
  numroc_(const int *n,
          const int *nb,
          const int *iproc,
          const int *isproc,
          const int *nprocs);

  void
  pdpotrf_(const char *UPLO,
           const int * N,
           double *    A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pspotrf_(const char *UPLO,
           const int * N,
           float *     A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pdgetrf_(const int *m,
           const int *n,
           double *   A,
           const int *IA,
           const int *JA,
           const int *DESCA,
           int *      ipiv,
           int *      INFO);

  void
  psgetrf_(const int *m,
           const int *n,
           float *    A,
           const int *IA,
           const int *JA,
           const int *DESCA,
           int *      ipiv,
           int *      INFO);

  void
  pdpotri_(const char *UPLO,
           const int * N,
           double *    A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pspotri_(const char *UPLO,
           const int * N,
           float *     A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pdgetri_(const int *N,
           double *   A,
           const int *IA,
           const int *JA,
           const int *DESCA,
           const int *ipiv,
           double *   work,
           int *      lwork,
           int *      iwork,
           int *      liwork,
           int *      info);

  void
  psgetri_(const int *N,
           float *    A,
           const int *IA,
           const int *JA,
           const int *DESCA,
           const int *ipiv,
           float *    work,
           int *      lwork,
           int *      iwork,
           int *      liwork,
           int *      info);

  void
  pdtrtri_(const char *UPLO,
           const char *DIAG,
           const int * N,
           double *    A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pstrtri_(const char *UPLO,
           const char *DIAG,
           const int * N,
           float *     A,
           const int * IA,
           const int * JA,
           const int * DESCA,
           int *       INFO);

  void
  pdpocon_(const char *  uplo,
           const int *   N,
           const double *A,
           const int *   IA,
           const int *   JA,
           const int *   DESCA,
           const double *ANORM,
           double *      RCOND,
           double *      WORK,
           const int *   LWORK,
           int *         IWORK,
           const int *   LIWORK,
           int *         INFO);

  void
  pspocon_(const char * uplo,
           const int *  N,
           const float *A,
           const int *  IA,
           const int *  JA,
           const int *  DESCA,
           const float *ANORM,
           float *      RCOND,
           float *      WORK,
           const int *  LWORK,
           int *        IWORK,
           const int *  LIWORK,
           int *        INFO);

  double
  pdlansy_(const char *  norm,
           const char *  uplo,
           const int *   N,
           const double *A,
           const int *   IA,
           const int *   JA,
           const int *   DESCA,
           double *      work);

  float
  pslansy_(const char * norm,
           const char * uplo,
           const int *  N,
           const float *A,
           const int *  IA,
           const int *  JA,
           const int *  DESCA,
           float *      work);

  int
  ilcm_(const int *M, const int *N);

  int
  iceil_(const int *i1, const int *i2);

  void
  descinit_(int *      desc,
            const int *m,
            const int *n,
            const int *mb,
            const int *nb,
            const int *irsrc,
            const int *icsrc,
            const int *ictxt,
            const int *lld,
            int *      info);

  int
  indxl2g_(const int *indxloc,
           const int *nb,
           const int *iproc,
           const int *isrcproc,
           const int *nprocs);

  void
  pdgesv_(const int *n,
          const int *nrhs,
          double *   A,
          const int *ia,
          const int *ja,
          const int *desca,
          int *      ipiv,
          double *   B,
          const int *ib,
          const int *jb,
          const int *descb,
          int *      info);

  void
  psgesv_(const int *n,
          const int *nrhs,
          float *    A,
          const int *ia,
          const int *ja,
          const int *desca,
          int *      ipiv,
          float *    B,
          const int *ib,
          const int *jb,
          const int *descb,
          int *      info);

  void
  pdgemm_(const char *  transa,
          const char *  transb,
          const int *   m,
          const int *   n,
          const int *   k,
          const double *alpha,
          const double *A,
          const int *   IA,
          const int *   JA,
          const int *   DESCA,
          const double *B,
          const int *   IB,
          const int *   JB,
          const int *   DESCB,
          const double *beta,
          double *      C,
          const int *   IC,
          const int *   JC,
          const int *   DESCC);

  void
  psgemm_(const char * transa,
          const char * transb,
          const int *  m,
          const int *  n,
          const int *  k,
          const float *alpha,
          const float *A,
          const int *  IA,
          const int *  JA,
          const int *  DESCA,
          const float *B,
          const int *  IB,
          const int *  JB,
          const int *  DESCB,
          const float *beta,
          float *      C,
          const int *  IC,
          const int *  JC,
          const int *  DESCC);

  double
  pdlange_(char const *  norm,
           const int *   m,
           const int *   n,
           const double *A,
           const int *   ia,
           const int *   ja,
           const int *   desca,
           double *      work);

  float
  pslange_(const char * norm,
           const int *  m,
           const int *  n,
           const float *A,
           const int *  ia,
           const int *  ja,
           const int *  desca,
           float *      work);

  int
  indxg2p_(const int *glob,
           const int *nb,
           const int *iproc,
           const int *isproc,
           const int *nprocs);

  void
  pdsyev_(const char *jobz,
          const char *uplo,
          const int * m,
          double *    A,
          const int * ia,
          const int * ja,
          int *       desca,
          double *    w,
          double *    z,
          const int * iz,
          const int * jz,
          int *       descz,
          double *    work,
          const int * lwork,
          int *       info);

  void
  pssyev_(const char *jobz,
          const char *uplo,
          const int * m,
          float *     A,
          const int * ia,
          const int * ja,
          int *       desca,
          float *     w,
          float *     z,
          const int * iz,
          const int * jz,
          int *       descz,
          float *     work,
          const int * lwork,
          int *       info);

  void
  pdlacpy_(const char *  uplo,
           const int *   m,
           const int *   n,
           const double *A,
           const int *   ia,
           const int *   ja,
           const int *   desca,
           double *      B,
           const int *   ib,
           const int *   jb,
           const int *   descb);

  void
  pslacpy_(const char * uplo,
           const int *  m,
           const int *  n,
           const float *A,
           const int *  ia,
           const int *  ja,
           const int *  desca,
           float *      B,
           const int *  ib,
           const int *  jb,
           const int *  descb);

  void
  pdgemr2d_(const int *   m,
            const int *   n,
            const double *A,
            const int *   ia,
            const int *   ja,
            const int *   desca,
            double *      B,
            const int *   ib,
            const int *   jb,
            const int *   descb,
            const int *   ictxt);

  void
  psgemr2d_(const int *  m,
            const int *  n,
            const float *A,
            const int *  ia,
            const int *  ja,
            const int *  desca,
            float *      B,
            const int *  ib,
            const int *  jb,
            const int *  descb,
            const int *  ictxt);

  double
  pdlamch_(const int *ictxt, const char *cmach);

  float
  pslamch_(const int *ictxt, const char *cmach);

  void
  pdsyevx_(const char *  jobz,
           const char *  range,
           const char *  uplo,
           const int *   n,
           double *      A,
           const int *   ia,
           const int *   ja,
           const int *   desca,
           const double *VL,
           const double *VU,
           const int *   il,
           const int *   iu,
           const double *abstol,
           const int *   m,
           const int *   nz,
           double *      w,
           double *      orfac,
           double *      Z,
           const int *   iz,
           const int *   jz,
           const int *   descz,
           double *      work,
           int *         lwork,
           int *         iwork,
           int *         liwork,
           int *         ifail,
           int *         iclustr,
           double *      gap,
           int *         info);

  void
  pssyevx_(const char * jobz,
           const char * range,
           const char * uplo,
           const int *  n,
           float *      A,
           const int *  ia,
           const int *  ja,
           const int *  desca,
           const float *VL,
           const float *VU,
           const int *  il,
           const int *  iu,
           const float *abstol,
           const int *  m,
           const int *  nz,
           float *      w,
           float *      orfac,
           float *      Z,
           const int *  iz,
           const int *  jz,
           const int *  descz,
           float *      work,
           int *        lwork,
           int *        iwork,
           int *        liwork,
           int *        ifail,
           int *        iclustr,
           float *      gap,
           int *        info);

  void
  pdgesvd_(const char *jobu,
           const char *jobvt,
           const int * m,
           const int * n,
           double *    A,
           const int * ia,
           const int * ja,
           const int * desca,
           double *    S,
           double *    U,
           const int * iu,
           const int * ju,
           const int * descu,
           double *    VT,
           const int * ivt,
           const int * jvt,
           const int * descvt,
           double *    work,
           int *       lwork,
           int *       info);

  void
  psgesvd_(const char *jobu,
           const char *jobvt,
           const int * m,
           const int * n,
           float *     A,
           const int * ia,
           const int * ja,
           const int * desca,
           float *     S,
           float *     U,
           const int * iu,
           const int * ju,
           const int * descu,
           float *     VT,
           const int * ivt,
           const int * jvt,
           const int * descvt,
           float *     work,
           int *       lwork,
           int *       info);

  void
  pdgels_(const char *trans,
          const int * m,
          const int * n,
          const int * nrhs,
          double *    A,
          const int * ia,
          const int * ja,
          const int * desca,
          double *    B,
          const int * ib,
          const int * jb,
          const int * descb,
          double *    work,
          int *       lwork,
          int *       info);

  void
  psgels_(const char *trans,
          const int * m,
          const int * n,
          const int * nrhs,
          float *     A,
          const int * ia,
          const int * ja,
          const int * desca,
          float *     B,
          const int * ib,
          const int * jb,
          const int * descb,
          float *     work,
          int *       lwork,
          int *       info);

  void
  pdgeadd_(const char *  transa,
           const int *   m,
           const int *   n,
           const double *alpha,
           const double *A,
           const int *   IA,
           const int *   JA,
           const int *   DESCA,
           const double *beta,
           double *      C,
           const int *   IC,
           const int *   JC,
           const int *   DESCC);

  void
  psgeadd_(const char * transa,
           const int *  m,
           const int *  n,
           const float *alpha,
           const float *A,
           const int *  IA,
           const int *  JA,
           const int *  DESCA,
           const float *beta,
           float *      C,
           const int *  IC,
           const int *  JC,
           const int *  DESCC);

  void
  pdtran_(const int *   m,
          const int *   n,
          const double *alpha,
          const double *A,
          const int *   IA,
          const int *   JA,
          const int *   DESCA,
          const double *beta,
          double *      C,
          const int *   IC,
          const int *   JC,
          const int *   DESCC);

  void
  pstran_(const int *  m,
          const int *  n,
          const float *alpha,
          const float *A,
          const int *  IA,
          const int *  JA,
          const int *  DESCA,
          const float *beta,
          float *      C,
          const int *  IC,
          const int *  JC,
          const int *  DESCC);

  void
  pdsyevr_(const char *  jobz,
           const char *  range,
           const char *  uplo,
           const int *   n,
           double *      A,
           const int *   IA,
           const int *   JA,
           const int *   DESCA,
           const double *VL,
           const double *VU,
           const int *   IL,
           const int *   IU,
           int *         m,
           int *         nz,
           double *      w,
           double *      Z,
           const int *   IZ,
           const int *   JZ,
           const int *   DESCZ,
           double *      work,
           int *         lwork,
           int *         iwork,
           int *         liwork,
           int *         info);

  void
  pssyevr_(const char * jobz,
           const char * range,
           const char * uplo,
           const int *  n,
           float *      A,
           const int *  IA,
           const int *  JA,
           const int *  DESCA,
           const float *VL,
           const float *VU,
           const int *  IL,
           const int *  IU,
           int *        m,
           int *        nz,
           float *      w,
           float *      Z,
           const int *  IZ,
           const int *  JZ,
           const int *  DESCZ,
           float *      work,
           int *        lwork,
           int *        iwork,
           int *        liwork,
           int *        info);



  //-----------------------------------------------------------------------------------------------------------------------------------------------------------

  void pdormqr(const char *  side,
			const char *  trans,
			const int *   m,
			const int *   n,
			const int *   k,
			double *A,
			const int *   IA,
			const int *   JA,
			const int *   DESCA,
			const double * tau,
			double *      C,
			const int *   IC,
			const int *   JC,
			const int *   DESCC,
			double *      work,
			int *         lwork,
			int *         info  );

  void psormqr(const char *  side,
  			const char *  trans,
  			const int *   m,
  			const int *   n,
  			const int *   k,
  			 float *A,
  			const int *   IA,
  			const int *   JA,
  			const int *   DESCA,
  			const float * tau,
			float *      C,
  			const int *   IC,
  			const int *   JC,
  			const int *   DESCC,
			float *      work,
  			int *         lwork,
  			int *         info  );


  void pdtrsv_(const char * uplo,
			const char * trans,
			const char * diag,
			const int * n,
			const double * A,
			const int * IA,
			const int * JA,
			const int * DESCA,
			double * X,
			const int * IX,
			const int * JX,
			const int * DESCX,
			const int * incx  );

  void pstrsv_(const char * uplo,
  			const char * trans,
  			const char * diag,
  			const int * n,
  			const float * A,
  			const int * IA,
  			const int * JA,
  			const int * DESCA,
			float * X,
  			const int * IX,
  			const int * JX,
  			const int * DESCX,
  			const int * incx  );


  void pdgeqrf(const int *   m,
			const int *   n,
			double *A,
			const int *   IA,
			const int *   JA,
			const int *   DESCA,
			double * tau,
			double *      work,
			int *         lwork,
			int *         info );

  void psgeqrf(const int *   m,
  			const int *   n,
  			float *A,
  			const int *   IA,
  			const int *   JA,
  			const int *   DESCA,
  			float * tau,
			float *      work,
  			int *         lwork,
  			int *         info );

  //------------------------------------------------------------------------------------------------------------------------------------------------------------
}



template <typename number>
inline void
Cgerv2d(int /*context*/,
        int /*M*/,
        int /*N*/,
        number * /*A*/,
        int /*lda*/,
        int /*rsrc*/,
        int /*csrc*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
Cgerv2d(int context, int M, int N, double *A, int lda, int rsrc, int csrc)
{
  Cdgerv2d(context, M, N, A, lda, rsrc, csrc);
}

inline void
Cgerv2d(int context, int M, int N, float *A, int lda, int rsrc, int csrc)
{
  Csgerv2d(context, M, N, A, lda, rsrc, csrc);
}


template <typename number>
inline void
Cgesd2d(int /*context*/,
        int /*M*/,
        int /*N*/,
        number * /*A*/,
        int /*lda*/,
        int /*rdest*/,
        int /*cdest*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
Cgesd2d(int context, int M, int N, double *A, int lda, int rdest, int cdest)
{
  Cdgesd2d(context, M, N, A, lda, rdest, cdest);
}

inline void
Cgesd2d(int context, int M, int N, float *A, int lda, int rdest, int cdest)
{
  Csgesd2d(context, M, N, A, lda, rdest, cdest);
}


template <typename number>
inline void
ppotrf(const char * /*UPLO*/,
       const int * /*N*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       int * /*INFO*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ppotrf(const char *UPLO,
       const int * N,
       double *    A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pdpotrf_(UPLO, N, A, IA, JA, DESCA, INFO);
}

inline void
ppotrf(const char *UPLO,
       const int * N,
       float *     A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pspotrf_(UPLO, N, A, IA, JA, DESCA, INFO);
}


template <typename number>
inline void
pgetrf(const int * /*m*/,
       const int * /*n*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       int * /*ipiv*/,
       int * /*INFO*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgetrf(const int *m,
       const int *n,
       double *   A,
       const int *IA,
       const int *JA,
       const int *DESCA,
       int *      ipiv,
       int *      INFO)
{
  pdgetrf_(m, n, A, IA, JA, DESCA, ipiv, INFO);
}

inline void
pgetrf(const int *m,
       const int *n,
       float *    A,
       const int *IA,
       const int *JA,
       const int *DESCA,
       int *      ipiv,
       int *      INFO)
{
  psgetrf_(m, n, A, IA, JA, DESCA, ipiv, INFO);
}


template <typename number>
inline void
ppotri(const char * /*UPLO*/,
       const int * /*N*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       int * /*INFO*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ppotri(const char *UPLO,
       const int * N,
       double *    A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pdpotri_(UPLO, N, A, IA, JA, DESCA, INFO);
}

inline void
ppotri(const char *UPLO,
       const int * N,
       float *     A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pspotri_(UPLO, N, A, IA, JA, DESCA, INFO);
}


template <typename number>
inline void
pgetri(const int * /*N*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       const int * /*ipiv*/,
       number * /*work*/,
       int * /*lwork*/,
       int * /*iwork*/,
       int * /*liwork*/,
       int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgetri(const int *N,
       double *   A,
       const int *IA,
       const int *JA,
       const int *DESCA,
       const int *ipiv,
       double *   work,
       int *      lwork,
       int *      iwork,
       int *      liwork,
       int *      info)
{
  pdgetri_(N, A, IA, JA, DESCA, ipiv, work, lwork, iwork, liwork, info);
}

inline void
pgetri(const int *N,
       float *    A,
       const int *IA,
       const int *JA,
       const int *DESCA,
       const int *ipiv,
       float *    work,
       int *      lwork,
       int *      iwork,
       int *      liwork,
       int *      info)
{
  psgetri_(N, A, IA, JA, DESCA, ipiv, work, lwork, iwork, liwork, info);
}

template <typename number>
inline void
ptrtri(const char * /*UPLO*/,
       const char * /*DIAG*/,
       const int * /*N*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       int * /*INFO*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ptrtri(const char *UPLO,
       const char *DIAG,
       const int * N,
       double *    A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pdtrtri_(UPLO, DIAG, N, A, IA, JA, DESCA, INFO);
}

inline void
ptrtri(const char *UPLO,
       const char *DIAG,
       const int * N,
       float *     A,
       const int * IA,
       const int * JA,
       const int * DESCA,
       int *       INFO)
{
  pstrtri_(UPLO, DIAG, N, A, IA, JA, DESCA, INFO);
}

template <typename number>
inline void
ppocon(const char * /*uplo*/,
       const int * /*N*/,
       const number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       const number * /*ANORM*/,
       number * /*RCOND*/,
       number * /*WORK*/,
       const int * /*LWORK*/,
       int * /*IWORK*/,
       const int * /*LIWORK*/,
       int * /*INFO*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ppocon(const char *  uplo,
       const int *   N,
       const double *A,
       const int *   IA,
       const int *   JA,
       const int *   DESCA,
       const double *ANORM,
       double *      RCOND,
       double *      WORK,
       const int *   LWORK,
       int *         IWORK,
       const int *   LIWORK,
       int *         INFO)
{
  pdpocon_(
    uplo, N, A, IA, JA, DESCA, ANORM, RCOND, WORK, LWORK, IWORK, LIWORK, INFO);
}

inline void
ppocon(const char * uplo,
       const int *  N,
       const float *A,
       const int *  IA,
       const int *  JA,
       const int *  DESCA,
       const float *ANORM,
       float *      RCOND,
       float *      WORK,
       const int *  LWORK,
       int *        IWORK,
       const int *  LIWORK,
       int *        INFO)
{
  pspocon_(
    uplo, N, A, IA, JA, DESCA, ANORM, RCOND, WORK, LWORK, IWORK, LIWORK, INFO);
}


template <typename number>
inline number
plansy(const char * /*norm*/,
       const char * /*uplo*/,
       const int * /*N*/,
       const number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       number * /*work*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline double
plansy(const char *  norm,
       const char *  uplo,
       const int *   N,
       const double *A,
       const int *   IA,
       const int *   JA,
       const int *   DESCA,
       double *      work)
{
  return pdlansy_(norm, uplo, N, A, IA, JA, DESCA, work);
}

inline float
plansy(const char * norm,
       const char * uplo,
       const int *  N,
       const float *A,
       const int *  IA,
       const int *  JA,
       const int *  DESCA,
       float *      work)
{
  return pslansy_(norm, uplo, N, A, IA, JA, DESCA, work);
}


template <typename number>
inline void
pgesv(const int * /*n*/,
      const int * /*nrhs*/,
      number * /*A*/,
      const int * /*ia*/,
      const int * /*ja*/,
      const int * /*desca*/,
      int * /*ipiv*/,
      number * /*B*/,
      const int * /*ib*/,
      const int * /*jb*/,
      const int * /*descb*/,
      int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgesv(const int *n,
      const int *nrhs,
      double *   A,
      const int *ia,
      const int *ja,
      const int *desca,
      int *      ipiv,
      double *   B,
      const int *ib,
      const int *jb,
      const int *descb,
      int *      info)
{
  pdgesv_(n, nrhs, A, ia, ja, desca, ipiv, B, ib, jb, descb, info);
}

inline void
pgesv(const int *n,
      const int *nrhs,
      float *    A,
      const int *ia,
      const int *ja,
      const int *desca,
      int *      ipiv,
      float *    B,
      const int *ib,
      const int *jb,
      const int *descb,
      int *      info)
{
  psgesv_(n, nrhs, A, ia, ja, desca, ipiv, B, ib, jb, descb, info);
}


template <typename number>
inline void
pgemm(const char * /*transa*/,
      const char * /*transb*/,
      const int * /*m*/,
      const int * /*n*/,
      const int * /*k*/,
      const number * /*alpha*/,
      number * /*A*/,
      const int * /*IA*/,
      const int * /*JA*/,
      const int * /*DESCA*/,
      number * /*B*/,
      const int * /*IB*/,
      const int * /*JB*/,
      const int * /*DESCB*/,
      const number * /*beta*/,
      number * /*C*/,
      const int * /*IC*/,
      const int * /*JC*/,
      const int * /*DESCC*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgemm(const char *  transa,
      const char *  transb,
      const int *   m,
      const int *   n,
      const int *   k,
      const double *alpha,
      const double *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
      const double *B,
      const int *   IB,
      const int *   JB,
      const int *   DESCB,
      const double *beta,
      double *      C,
      const int *   IC,
      const int *   JC,
      const int *   DESCC)
{
  pdgemm_(transa,
          transb,
          m,
          n,
          k,
          alpha,
          A,
          IA,
          JA,
          DESCA,
          B,
          IB,
          JB,
          DESCB,
          beta,
          C,
          IC,
          JC,
          DESCC);
}

inline void
pgemm(const char * transa,
      const char * transb,
      const int *  m,
      const int *  n,
      const int *  k,
      const float *alpha,
      const float *A,
      const int *  IA,
      const int *  JA,
      const int *  DESCA,
      const float *B,
      const int *  IB,
      const int *  JB,
      const int *  DESCB,
      const float *beta,
      float *      C,
      const int *  IC,
      const int *  JC,
      const int *  DESCC)
{
  psgemm_(transa,
          transb,
          m,
          n,
          k,
          alpha,
          A,
          IA,
          JA,
          DESCA,
          B,
          IB,
          JB,
          DESCB,
          beta,
          C,
          IC,
          JC,
          DESCC);
}


template <typename number>
inline number
plange(const char * /*norm*/,
       const int * /*m*/,
       const int * /*n*/,
       const number * /*A*/,
       const int * /*ia*/,
       const int * /*ja*/,
       const int * /*desca*/,
       number * /*work*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline double
plange(const char *  norm,
       const int *   m,
       const int *   n,
       const double *A,
       const int *   ia,
       const int *   ja,
       const int *   desca,
       double *      work)
{
  return pdlange_(norm, m, n, A, ia, ja, desca, work);
}

inline float
plange(const char * norm,
       const int *  m,
       const int *  n,
       const float *A,
       const int *  ia,
       const int *  ja,
       const int *  desca,
       float *      work)
{
  return pslange_(norm, m, n, A, ia, ja, desca, work);
}


template <typename number>
inline void
psyev(const char * /*jobz*/,
      const char * /*uplo*/,
      const int * /*m*/,
      number * /*A*/,
      const int * /*ia*/,
      const int * /*ja*/,
      int * /*desca*/,
      number * /*w*/,
      number * /*z*/,
      const int * /*iz*/,
      const int * /*jz*/,
      int * /*descz*/,
      number * /*work*/,
      const int * /*lwork*/,
      int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
psyev(const char *jobz,
      const char *uplo,
      const int * m,
      double *    A,
      const int * ia,
      const int * ja,
      int *       desca,
      double *    w,
      double *    z,
      const int * iz,
      const int * jz,
      int *       descz,
      double *    work,
      const int * lwork,
      int *       info)
{
  pdsyev_(
    jobz, uplo, m, A, ia, ja, desca, w, z, iz, jz, descz, work, lwork, info);
}

inline void
psyev(const char *jobz,
      const char *uplo,
      const int * m,
      float *     A,
      const int * ia,
      const int * ja,
      int *       desca,
      float *     w,
      float *     z,
      const int * iz,
      const int * jz,
      int *       descz,
      float *     work,
      const int * lwork,
      int *       info)
{
  pssyev_(
    jobz, uplo, m, A, ia, ja, desca, w, z, iz, jz, descz, work, lwork, info);
}


template <typename number>
inline void
placpy(const char * /*uplo*/,
       const int * /*m*/,
       const int * /*n*/,
       const number * /*A*/,
       const int * /*ia*/,
       const int * /*ja*/,
       const int * /*desca*/,
       number * /*B*/,
       const int * /*ib*/,
       const int * /*jb*/,
       const int * /*descb*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
placpy(const char *  uplo,
       const int *   m,
       const int *   n,
       const double *A,
       const int *   ia,
       const int *   ja,
       const int *   desca,
       double *      B,
       const int *   ib,
       const int *   jb,
       const int *   descb)
{
  pdlacpy_(uplo, m, n, A, ia, ja, desca, B, ib, jb, descb);
}

inline void
placpy(const char * uplo,
       const int *  m,
       const int *  n,
       const float *A,
       const int *  ia,
       const int *  ja,
       const int *  desca,
       float *      B,
       const int *  ib,
       const int *  jb,
       const int *  descb)
{
  pslacpy_(uplo, m, n, A, ia, ja, desca, B, ib, jb, descb);
}


template <typename number>
inline void
pgemr2d(const int * /*m*/,
        const int * /*n*/,
        const number * /*A*/,
        const int * /*ia*/,
        const int * /*ja*/,
        const int * /*desca*/,
        number * /*B*/,
        const int * /*ib*/,
        const int * /*jb*/,
        const int * /*descb*/,
        const int * /*ictxt*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgemr2d(const int *   m,
        const int *   n,
        const double *A,
        const int *   ia,
        const int *   ja,
        const int *   desca,
        double *      B,
        const int *   ib,
        const int *   jb,
        const int *   descb,
        const int *   ictxt)
{
  pdgemr2d_(m, n, A, ia, ja, desca, B, ib, jb, descb, ictxt);
}

inline void
pgemr2d(const int *  m,
        const int *  n,
        const float *A,
        const int *  ia,
        const int *  ja,
        const int *  desca,
        float *      B,
        const int *  ib,
        const int *  jb,
        const int *  descb,
        const int *  ictxt)
{
  psgemr2d_(m, n, A, ia, ja, desca, B, ib, jb, descb, ictxt);
}


template <typename number>
inline void
plamch(const int * /*ictxt*/, const char * /*cmach*/, number & /*val*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
plamch(const int *ictxt, const char *cmach, double &val)
{
  val = pdlamch_(ictxt, cmach);
}

inline void
plamch(const int *ictxt, const char *cmach, float &val)
{
  val = pslamch_(ictxt, cmach);
}


template <typename number>
inline void
psyevx(const char * /*jobz*/,
       const char * /*range*/,
       const char * /*uplo*/,
       const int * /*n*/,
       number * /*A*/,
       const int * /*ia*/,
       const int * /*ja*/,
       const int * /*desca*/,
       number * /*VL*/,
       number * /*VU*/,
       const int * /*il*/,
       const int * /*iu*/,
       number * /*abstol*/,
       const int * /*m*/,
       const int * /*nz*/,
       number * /*w*/,
       number * /*orfac*/,
       number * /*Z*/,
       const int * /*iz*/,
       const int * /*jz*/,
       const int * /*descz*/,
       number * /*work*/,
       int * /*lwork*/,
       int * /*iwork*/,
       int * /*liwork*/,
       int * /*ifail*/,
       int * /*iclustr*/,
       number * /*gap*/,
       int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
psyevx(const char *jobz,
       const char *range,
       const char *uplo,
       const int * n,
       double *    A,
       const int * ia,
       const int * ja,
       const int * desca,
       double *    VL,
       double *    VU,
       const int * il,
       const int * iu,
       double *    abstol,
       const int * m,
       const int * nz,
       double *    w,
       double *    orfac,
       double *    Z,
       const int * iz,
       const int * jz,
       const int * descz,
       double *    work,
       int *       lwork,
       int *       iwork,
       int *       liwork,
       int *       ifail,
       int *       iclustr,
       double *    gap,
       int *       info)
{
  pdsyevx_(jobz,
           range,
           uplo,
           n,
           A,
           ia,
           ja,
           desca,
           VL,
           VU,
           il,
           iu,
           abstol,
           m,
           nz,
           w,
           orfac,
           Z,
           iz,
           jz,
           descz,
           work,
           lwork,
           iwork,
           liwork,
           ifail,
           iclustr,
           gap,
           info);
}

inline void
psyevx(const char *jobz,
       const char *range,
       const char *uplo,
       const int * n,
       float *     A,
       const int * ia,
       const int * ja,
       const int * desca,
       float *     VL,
       float *     VU,
       const int * il,
       const int * iu,
       float *     abstol,
       const int * m,
       const int * nz,
       float *     w,
       float *     orfac,
       float *     Z,
       const int * iz,
       const int * jz,
       const int * descz,
       float *     work,
       int *       lwork,
       int *       iwork,
       int *       liwork,
       int *       ifail,
       int *       iclustr,
       float *     gap,
       int *       info)
{
  pssyevx_(jobz,
           range,
           uplo,
           n,
           A,
           ia,
           ja,
           desca,
           VL,
           VU,
           il,
           iu,
           abstol,
           m,
           nz,
           w,
           orfac,
           Z,
           iz,
           jz,
           descz,
           work,
           lwork,
           iwork,
           liwork,
           ifail,
           iclustr,
           gap,
           info);
}


template <typename number>
inline void
pgesvd(const char * /*jobu*/,
       const char * /*jobvt*/,
       const int * /*m*/,
       const int * /*n*/,
       number * /*A*/,
       const int * /*ia*/,
       const int * /*ja*/,
       const int * /*desca*/,
       number * /*S*/,
       number * /*U*/,
       const int * /*iu*/,
       const int * /*ju*/,
       const int * /*descu*/,
       number * /*VT*/,
       const int * /*ivt*/,
       const int * /*jvt*/,
       const int * /*descvt*/,
       number * /*work*/,
       int * /*lwork*/,
       int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgesvd(const char *jobu,
       const char *jobvt,
       const int * m,
       const int * n,
       double *    A,
       const int * ia,
       const int * ja,
       const int * desca,
       double *    S,
       double *    U,
       const int * iu,
       const int * ju,
       const int * descu,
       double *    VT,
       const int * ivt,
       const int * jvt,
       const int * descvt,
       double *    work,
       int *       lwork,
       int *       info)
{
  pdgesvd_(jobu,
           jobvt,
           m,
           n,
           A,
           ia,
           ja,
           desca,
           S,
           U,
           iu,
           ju,
           descu,
           VT,
           ivt,
           jvt,
           descvt,
           work,
           lwork,
           info);
}

inline void
pgesvd(const char *jobu,
       const char *jobvt,
       const int * m,
       const int * n,
       float *     A,
       const int * ia,
       const int * ja,
       const int * desca,
       float *     S,
       float *     U,
       const int * iu,
       const int * ju,
       const int * descu,
       float *     VT,
       const int * ivt,
       const int * jvt,
       const int * descvt,
       float *     work,
       int *       lwork,
       int *       info)
{
  psgesvd_(jobu,
           jobvt,
           m,
           n,
           A,
           ia,
           ja,
           desca,
           S,
           U,
           iu,
           ju,
           descu,
           VT,
           ivt,
           jvt,
           descvt,
           work,
           lwork,
           info);
}


template <typename number>
inline void
pgels(const char * /*trans*/,
      const int * /*m*/,
      const int * /*n*/,
      const int * /*nrhs*/,
      number * /*A*/,
      const int * /*ia*/,
      const int * /*ja*/,
      const int * /*desca*/,
      number * /*B*/,
      const int * /*ib*/,
      const int * /*jb*/,
      const int * /*descb*/,
      number * /*work*/,
      int * /*lwork*/,
      int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgels(const char *trans,
      const int * m,
      const int * n,
      const int * nrhs,
      double *    A,
      const int * ia,
      const int * ja,
      const int * desca,
      double *    B,
      const int * ib,
      const int * jb,
      const int * descb,
      double *    work,
      int *       lwork,
      int *       info)
{
  pdgels_(
    trans, m, n, nrhs, A, ia, ja, desca, B, ib, jb, descb, work, lwork, info);
}

inline void
pgels(const char *trans,
      const int * m,
      const int * n,
      const int * nrhs,
      float *     A,
      const int * ia,
      const int * ja,
      const int * desca,
      float *     B,
      const int * ib,
      const int * jb,
      const int * descb,
      float *     work,
      int *       lwork,
      int *       info)
{
  psgels_(
    trans, m, n, nrhs, A, ia, ja, desca, B, ib, jb, descb, work, lwork, info);
}


template <typename number>
inline void
pgeadd(const char * /*transa*/,
       const int * /*m*/,
       const int * /*n*/,
       const number * /*alpha*/,
       const number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       const number * /*beta*/,
       number * /*C*/,
       const int * /*IC*/,
       const int * /*JC*/,
       const int * /*DESCC*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgeadd(const char *  transa,
       const int *   m,
       const int *   n,
       const double *alpha,
       const double *A,
       const int *   IA,
       const int *   JA,
       const int *   DESCA,
       const double *beta,
       double *      C,
       const int *   IC,
       const int *   JC,
       const int *   DESCC)
{
  pdgeadd_(transa, m, n, alpha, A, IA, JA, DESCA, beta, C, IC, JC, DESCC);
}

inline void
pgeadd(const char * transa,
       const int *  m,
       const int *  n,
       const float *alpha,
       const float *A,
       const int *  IA,
       const int *  JA,
       const int *  DESCA,
       const float *beta,
       float *      C,
       const int *  IC,
       const int *  JC,
       const int *  DESCC)
{
  psgeadd_(transa, m, n, alpha, A, IA, JA, DESCA, beta, C, IC, JC, DESCC);
}


template <typename number>
inline void
ptran(const int * /*m*/,
      const int * /*n*/,
      const number * /*alpha*/,
      const number * /*A*/,
      const int * /*IA*/,
      const int * /*JA*/,
      const int * /*DESCA*/,
      const number * /*beta*/,
      number * /*C*/,
      const int * /*IC*/,
      const int * /*JC*/,
      const int * /*DESCC*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ptran(const int *   m,
      const int *   n,
      const double *alpha,
      const double *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
      const double *beta,
      double *      C,
      const int *   IC,
      const int *   JC,
      const int *   DESCC)
{
  pdtran_(m, n, alpha, A, IA, JA, DESCA, beta, C, IC, JC, DESCC);
}

inline void
ptran(const int *  m,
      const int *  n,
      const float *alpha,
      const float *A,
      const int *  IA,
      const int *  JA,
      const int *  DESCA,
      const float *beta,
      float *      C,
      const int *  IC,
      const int *  JC,
      const int *  DESCC)
{
  pstran_(m, n, alpha, A, IA, JA, DESCA, beta, C, IC, JC, DESCC);
}


template <typename number>
inline void
psyevr(const char * /*jobz*/,
       const char * /*range*/,
       const char * /*uplo*/,
       const int * /*n*/,
       number * /*A*/,
       const int * /*IA*/,
       const int * /*JA*/,
       const int * /*DESCA*/,
       const number * /*VL*/,
       const number * /*VU*/,
       const int * /*IL*/,
       const int * /*IU*/,
       int * /*m*/,
       int * /*nz*/,
       number * /*w*/,
       number * /*Z*/,
       const int * /*IZ*/,
       const int * /*JZ*/,
       const int * /*DESCZ*/,
       number * /*work*/,
       int * /*lwork*/,
       int * /*iwork*/,
       int * /*liwork*/,
       int * /*info*/)
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
psyevr(const char *  jobz,
       const char *  range,
       const char *  uplo,
       const int *   n,
       double *      A,
       const int *   IA,
       const int *   JA,
       const int *   DESCA,
       const double *VL,
       const double *VU,
       const int *   IL,
       const int *   IU,
       int *         m,
       int *         nz,
       double *      w,
       double *      Z,
       const int *   IZ,
       const int *   JZ,
       const int *   DESCZ,
       double *      work,
       int *         lwork,
       int *         iwork,
       int *         liwork,
       int *         info)
{
  pdsyevr_(jobz,
           range,
           uplo,
           n,
           A,
           IA,
           JA,
           DESCA,
           VL,
           VU,
           IL,
           IU,
           m,
           nz,
           w,
           Z,
           IZ,
           JZ,
           DESCZ,
           work,
           lwork,
           iwork,
           liwork,
           info);
}

inline void
psyevr(const char * jobz,
       const char * range,
       const char * uplo,
       const int *  n,
       float *      A,
       const int *  IA,
       const int *  JA,
       const int *  DESCA,
       const float *VL,
       const float *VU,
       const int *  IL,
       const int *  IU,
       int *        m,
       int *        nz,
       float *      w,
       float *      Z,
       const int *  IZ,
       const int *  JZ,
       const int *  DESCZ,
       float *      work,
       int *        lwork,
       int *        iwork,
       int *        liwork,
       int *        info)
{
  pssyevr_(jobz,
           range,
           uplo,
           n,
           A,
           IA,
           JA,
           DESCA,
           VL,
           VU,
           IL,
           IU,
           m,
           nz,
           w,
           Z,
           IZ,
           JZ,
           DESCZ,
           work,
           lwork,
           iwork,
           liwork,
           info);
}



//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

//PDORMQR-------------------------------------------------
template <typename number>
inline void
pormqr(const char *,
      const char * ,
      const int * ,
      const int * ,
      const int * ,
      number * ,
      const int * ,
      const int * ,
      const int * ,
	  const number * ,
      number * ,
      const int * ,
      const int * ,
      const int * ,
	  number * ,
	   int * ,
	  int *
      )
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pormqr(const char *  side,
      const char *  trans,
      const int *   m,
      const int *   n,
      const int *   k,
      double *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
	  const double * tau,
      double *      C,
      const int *   IC,
      const int *   JC,
      const int *   DESCC,
	  double *      work,
	   int *   lwork,
	  int *         info
	  )
{
  pdormqr(side,
          trans,
          m,
          n,
          k,
          A,
          IA,
          JA,
          DESCA,
          tau,
          C,
          IC,
          JC,
          DESCC,
		  work,
		  lwork,
		  info);
}


inline void
pormqr(const char *  side,
      const char *  trans,
      const int *   m,
      const int *   n,
      const int *   k,
      float *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
	  const float * tau,
      float *      C,
      const int *   IC,
      const int *   JC,
      const int *   DESCC,
	  float *      work,
	   int *   lwork,
	  int *         info
	  )
{
  psormqr(side,
          trans,
          m,
          n,
          k,
          A,
          IA,
          JA,
          DESCA,
          tau,
          C,
          IC,
          JC,
          DESCC,
		  work,
		  lwork,
		  info);
}


//PDTRSV_ -------------------------------------------------------------
template <typename number>
inline void
ptrsv(const char * ,
      const char * ,
	  const char * ,
      const int * ,
      const number * ,
      const int * ,
      const int * ,
      const int * ,
      number * ,
      const int * ,
      const int * ,
      const int * ,
	  const int *
      )
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
ptrsv(const char * uplo,
      const char * trans,
	  const char * diag,
      const int * n,
      const double * A,
      const int * IA,
      const int * JA,
      const int * DESCA,
      double * X,
      const int * IX,
      const int * JX,
      const int * DESCX,
	  const int * incx
      )
{
  pdtrsv_(uplo,
          trans,
		  diag,
          n,
          A,
          IA,
          JA,
          DESCA,
          X,
          IX,
          JX,
          DESCX,
		  incx);
}


inline void
ptrsv(const char * uplo,
      const char * trans,
	  const char * diag,
      const int * n,
      const float * A,
      const int * IA,
      const int * JA,
      const int * DESCA,
      float * X,
      const int * IX,
      const int * JX,
      const int * DESCX,
	  const int * incx
      )
{
  pstrsv_(uplo,
          trans,
		  diag,
          n,
          A,
          IA,
          JA,
          DESCA,
          X,
          IX,
          JX,
          DESCX,
		  incx);
}



//PDGEQRF ----------------------------------
template <typename number>
inline void
pgeqrf(const int * ,
      const int * ,
      number * ,
      const int * ,
      const int * ,
      const int * ,
	  number * ,
	  number * ,
	  int * ,
	  int *
      )
{
  Assert(false, dealii::ExcNotImplemented());
}

inline void
pgeqrf(const int *   m,
      const int *   n,
      double *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
	  double * tau,
	  double *      work,
	  int *         lwork,
	  int *         info
	  )
{
  pdgeqrf(m,
          n,
          A,
          IA,
          JA,
          DESCA,
          tau,
		  work,
		  lwork,
		  info);
}


inline void
pgeqrf(const int *   m,
      const int *   n,
      float *A,
      const int *   IA,
      const int *   JA,
      const int *   DESCA,
	  float * tau,
	  float *      work,
	  int *         lwork,
	  int *         info
	  )
{
  psgeqrf(m,
          n,
          A,
          IA,
          JA,
          DESCA,
          tau,
		  work,
		  lwork,
		  info);
}





#endif // scalapack_templates_h
