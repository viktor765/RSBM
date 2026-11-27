#!/usr/bin/env bash
# Download the paper's Zenodo record (doi:10.5281/zenodo.24574007) into the
# layout of the training runs, checking every file against the checksums below.
#
# Usage, from the repository root: bash scripts/download_zenodo.sh [OUT_DIR [PATTERN ...]]
#   OUT_DIR  where the files go (default: outputs/zenodo)
#   PATTERN  only the weights whose record names match one of these globs, e.g.
#            'mnist_*' 'checkerboard_*' (default: all); the records always come
#
# training_records.zip is unzipped to OUT_DIR/training_records/, which holds
# every paper run as REPRODUCE.md's commands name it. Each weight file goes into
# its run there, as training writes it: <run>/checkpoints/final_model_ema.safetensors.
# Files already present with the right checksum are not downloaded again.
set -euo pipefail

DOI=10.5281/zenodo.24574007
OUT=${1:-outputs/zenodo}
if [[ $# -ge 2 ]]; then PATTERNS=("${@:2}"); else PATTERNS=('*'); fi
RUNS=training_records
WEIGHTS=checkpoints/final_model_ema.safetensors
# Checksum, name in the record, and run under training_records/ of each weight file.
FILES="\
c41fb269380012bb5a575e075044bed10ca5d7b88e042d00a311ee7fa4062c22  afhq64_alpha_imf_non-refl_seed0.safetensors  afhq64/alpha_imf/non-refl/sigma0.75/seed0
28f0015b0ff9ebda0eba6b70cb62495ff7709612abb455023c895cb76e23965c  afhq64_alpha_imf_non-refl_seed1.safetensors  afhq64/alpha_imf/non-refl/sigma0.75/seed1
017e388e9cfe777c79a75834700d774a00c33dd49c1a9a8f10709fd8f69104bc  afhq64_alpha_imf_non-refl_seed2.safetensors  afhq64/alpha_imf/non-refl/sigma0.75/seed2
66dcb51e2d97dc04ca92182d962d17abff4b06cd3d64f6d9baabee87209816de  afhq64_alpha_imf_refl_seed0.safetensors  afhq64/alpha_imf/refl/sigma0.75/seed0
2ae6371ff078088ed14a255d6128864b74390041a8b598801ff2faa4f3f0ea74  afhq64_alpha_imf_refl_seed1.safetensors  afhq64/alpha_imf/refl/sigma0.75/seed1
2fb6cec1b6d71d670a99b0c46346992fcb10bbcb0d17f9bd9d53f15650be3a57  afhq64_alpha_imf_refl_seed2.safetensors  afhq64/alpha_imf/refl/sigma0.75/seed2
963e2ec2a9663df38f206929bced433e7d1e64ab1df182b95825aad9bfd35d1b  checkerboard_alpha_imf_refl_seed0.safetensors  uniform-to-checkerboard/alpha_imf/refl/sigma0.5/seed0
5516e9ee434f84e20273f36627f4f98b784dca22b9ccef3bf135d7a4ae4d1359  checkerboard_alpha_imf_refl_seed1.safetensors  uniform-to-checkerboard/alpha_imf/refl/sigma0.5/seed1
92e80a487e5ff74482fef0da19399fa3c49f14fad0f507f01b4ef81b09a1d749  checkerboard_alpha_imf_refl_seed2.safetensors  uniform-to-checkerboard/alpha_imf/refl/sigma0.5/seed2
4b3bfbf47fbe6c31a630b57994abdaa91d0acc6e88c5856cac285057b1ba7f2c  checkerboard_rsb_fbsde_refl_seed0.safetensors  uniform-to-checkerboard/rsb_fbsde/refl/sigma0.5/seed0
8263ddda5f3c6f0a38990b41fca666533f28717a866c87c21a7e21ec914b65e8  checkerboard_rsb_fbsde_refl_seed1.safetensors  uniform-to-checkerboard/rsb_fbsde/refl/sigma0.5/seed1
e54c8aa1f2afc2946ffa36171c0ed75dc335bcd219cb38d65a0f4aeff458950b  checkerboard_rsb_fbsde_refl_seed2.safetensors  uniform-to-checkerboard/rsb_fbsde/refl/sigma0.5/seed2
7ce2d4ffc4d33e4f286c943aa9c9234fd7022f45c01aceafb79b8ca3c9bf9dee  mnist_to_emnist_alpha_imf_non-refl_seed0.safetensors  mnist-to-emnist/alpha_imf/non-refl/sigma1.0/seed0
ab76c37fc84e97612d5aaf91ce477a8afbf2a7bb0345a2b2dfcb82a929d1d635  mnist_to_emnist_alpha_imf_non-refl_seed1.safetensors  mnist-to-emnist/alpha_imf/non-refl/sigma1.0/seed1
afc0a7168cdd6fb61693c4788568cd1a1679beea935bea0f0d5a6f606be2126c  mnist_to_emnist_alpha_imf_non-refl_seed2.safetensors  mnist-to-emnist/alpha_imf/non-refl/sigma1.0/seed2
a6c89d7495242d29a71f3b9b254f50794f886b00769f74b49db45f9cf4bc0186  mnist_to_emnist_alpha_imf_refl_seed0.safetensors  mnist-to-emnist/alpha_imf/refl/sigma1.0/seed0
965d6e0063f9da756eb66089cd6808966290f201acc1392200322a0b4460ce23  mnist_to_emnist_alpha_imf_refl_seed1.safetensors  mnist-to-emnist/alpha_imf/refl/sigma1.0/seed1
de274bfeeee99282c303960785d85a2e56c21ad146814db8d9a092369ee8d548  mnist_to_emnist_alpha_imf_refl_seed2.safetensors  mnist-to-emnist/alpha_imf/refl/sigma1.0/seed2
6780cd55afafbfd9858e8f52e9d25d595d6022e6a49e6d8d0f13b1e20b164f54  oresund_alpha_imf_refl_seed0.safetensors  schrodinger-to-oresund/alpha_imf/refl/sigma0.5/seed0
7e45c7c949ff38cfeab2a180b1c90ecbfd99b8f5ecd6b8e83160e94740ab6eab  oresund_alpha_imf_refl_seed1.safetensors  schrodinger-to-oresund/alpha_imf/refl/sigma0.5/seed1
e9121ae3860d2beebd0ca8df5d0ed24bb5fa41cbd0c87ca44704023603dd92ab  oresund_alpha_imf_refl_seed2.safetensors  schrodinger-to-oresund/alpha_imf/refl/sigma0.5/seed2
8b1c0745f822d20b81bd3db35822bd02693780c9fcb802713725d0b2adc146c9  oresund_rsb_fbsde_refl_seed0.safetensors  schrodinger-to-oresund/rsb_fbsde/refl/sigma0.5/seed0
241f95efb2b0caa77d061ff16341a444865a23b0c79ecef399d39cbb854d657b  oresund_rsb_fbsde_refl_seed1.safetensors  schrodinger-to-oresund/rsb_fbsde/refl/sigma0.5/seed1
83d1171ab4982c9e5b0ef0bd0de8617dc39e39029204aec109cbe068ec62443e  oresund_rsb_fbsde_refl_seed2.safetensors  schrodinger-to-oresund/rsb_fbsde/refl/sigma0.5/seed2"
RECORDS_SHA256=932489d7af8b41051f690ed9c00e917af79a665cd8d6c6106b3fa7c574458354

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

# Download the record's file name to path, unless path already has the checksum.
fetch() {
    local hash=$1 name=$2 path=$3
    if [[ -f $path && $(sha256 "$path") == "$hash" ]]; then
        echo "Have $path"
        return
    fi
    echo "Downloading $name to $path"
    mkdir -p "$(dirname "$path")"
    curl -fL --retry 3 -o "$path" "$URL/$name?download=1"
    if [[ $(sha256 "$path") != "$hash" ]]; then
        rm -f "$path"
        echo "Checksum mismatch: $name (deleted)" >&2
        exit 1
    fi
}

wanted() {
    local pattern
    for pattern in "${PATTERNS[@]}"; do
        if [[ $1 == $pattern ]]; then return 0; fi
    done
    return 1
}

# The DOI resolves to the record's current page; its files are under /files.
RECORD=$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://doi.org/$DOI")
RECORD=${RECORD%%\?*}
URL=${RECORD%/}/files
echo "Record: ${RECORD%/}"

mkdir -p "$OUT"
cd "$OUT"
fetch "$RECORDS_SHA256" training_records.zip training_records.zip
unzip -oq training_records.zip
while read -r hash name run; do
    if wanted "$name"; then
        fetch "$hash" "$name" "$RUNS/$run/$WEIGHTS"
    fi
done <<< "$FILES"
echo "Runs in $OUT/$RUNS/"
