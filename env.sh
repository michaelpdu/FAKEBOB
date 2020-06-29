# don't need to execute it now, due to all of environment variables have been added into ~/.bashrc

. ./kaldi_models/ivector_models/cmd.sh
. ./kaldi_models/ivector_models/path.sh

export LD_LIBRARY_PATH=\
${KALDI_ROOT}/src/lib:\
${KALDI_ROOT}/tools/openfst-1.6.7/lib:\
$LD_LIBRARY_PATH