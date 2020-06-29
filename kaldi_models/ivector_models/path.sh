export KALDI_ROOT=/data1/github/kaldi
export PATH=\
${KALDI_ROOT}/tools/openfst/bin:\
${KALDI_ROOT}/tools/sph2pipe_v2.5:\
$PWD/pre-models/:\
$PWD/pre-models/utils/:\
$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

export LC_ALL=C
