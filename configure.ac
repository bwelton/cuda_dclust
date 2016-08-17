# /* Copyright Benjamin Welton 2015 */

AC_PREREQ([2.63])
AC_INIT([cuda_dclust], [0.0.1], [welton@cs.wisc.edu])

# Get Parent Working Directory

MAIN_DIR="`pwd`"
AC_SUBST(MAIN_DIR)

AC_ARG_WITH(cudaIncludes,
            [AS_HELP_STRING([--cuda-include=OPTIONS],
                            [Include flags (-I) for building with cuda on this platform])],
            [CUDA_INCLUDES="${withval}"],
            [CUDA_INCLUDES="${CRAY_CUDATOOLKIT_INCLUDE_OPTS}"])

AC_ARG_WITH(cudaLinkOptions,
            [AS_HELP_STRING([--cuda-link=OPTIONS],
                            [Linker flags (-L) for building with cuda on this platform])],
            [CUDA_LINK_OPTS="${withval}"],
            [CUDA_LINK_OPTS="${CRAY_CUDATOOLKIT_POST_LINK_OPTS}"])

AC_ARG_WITH(boostIncludes,
            [AS_HELP_STRING([--boost-include=OPTIONS],
                            [Boost includes (-I) for this platform])],
            [BOOST_INCLUDES="${withval}"],
            [BOOST_INCLUDES="${BOOST_INC}"])

AC_ARG_WITH(boostLibs,
            [AS_HELP_STRING([--boost-libs=OPTIONS],
                            [Boost libs (-L) for this platform])],
            [BOOST_LIBS="${withval}"],
            [BOOST_LIBS="${BOOST_LIB}"])

# Checks for programs.
AC_PATH_PROG(NVCC, nvcc)
AC_SUBST(NVCC)

AC_PROG_CC(xlc_r cc gcc pgcc icc)
AC_PROG_CXX(xlC_r CC g++ pgCC icpc)
AC_LANG(C++)

MPI_INCLUDES=$MPICH_DIR
MPI_INCLUDES=${MPI_INCLUDES}/include
AC_SUBST(MPI_INCLUDES)

#CUDA_INCLUDES=$CRAY_CUDATOOLKIT_INCLUDE_OPTS
#CUDA_LINK_OPTS=$CRAY_CUDATOOLKIT_POST_LINK_OPTS
AC_SUBST(CUDA_INCLUDES)
AC_SUBST(CUDA_LINK_OPTS)

SO_OPTS="-shared -fPIC"
AC_SUBST(SO_OPTS)

PREFIX=$prefix
BINDIR=${PREFIX}/bin
LIBDIR=${PREFIX}/lib

#BOOST_INCLUDES=$BOOST_INC
#BOOST_LIBS=$BOOST_LIB

AS_MKDIR_P(${PREFIX})
AS_MKDIR_P(${PREFIX}/lib)
AS_MKDIR_P(${PREFIX}/bin)

AC_SUBST(BINDIR)
AC_SUBST(LIBDIR)
AC_SUBST(BOOST_INCLUDES)
AC_SUBST(BOOST_LIBS)


BUILDTIME="$(date '+%m.%d.%y-%H:%M')"
AC_SUBST(BUILDTIME)

AC_CONFIG_FILES([Makefile:Makefile.in src/dbscan/Makefile:src/dbscan/Makefile.in])

AC_OUTPUT
