#!/bin/bash
# Logging helpers: record downloads, extractions and pre-existing dirs; print a summary on exit.
LOG_DOWNLOADS=()
LOG_EXTRACTS=()
LOG_FAIL_DOWNLOADS=()
LOG_FAIL_EXTRACTS=()
mapfile -t BEFORE_DIRS < <(find datasets -type d 2>/dev/null || true)

wget() {
    local args=("$@") out="" url=""
    for ((i=0;i<${#args[@]};i++)); do
        if [ "${args[i]}" = "-O" ]; then out="${args[i+1]}"; fi
    done
    url="${args[0]}"
    echo " [log] downloading ${url} -> ${out:-stdout}"
    if command wget "$@"; then
        LOG_DOWNLOADS+=("${out:-$url}")
    else
        LOG_FAIL_DOWNLOADS+=("${out:-$url}")
        return 1
    fi
}

unzip() {
    local args=("$@") dest="" zipfile
    for ((i=0;i<${#args[@]};i++)); do
        if [ "${args[i]}" = "-d" ]; then dest="${args[i+1]}"; fi
    done
    zipfile="${args[$((${#args[@]}-1))]}"
    echo " [log] extracting ${zipfile} -> ${dest:-.}"
    if command unzip "$@"; then
        LOG_EXTRACTS+=("${zipfile} -> ${dest:-.}")
    else
        LOG_FAIL_EXTRACTS+=("${zipfile} -> ${dest:-.}")
        return 1
    fi
}

print_summary() {
    echo
    echo "Download/Extraction summary:"
    if [ ${#LOG_DOWNLOADS[@]} -gt 0 ]; then
        echo "  Downloads performed:"
        for i in "${LOG_DOWNLOADS[@]}"; do echo "    - $i"; done
    else
        echo "  No downloads performed."
    fi

    if [ ${#LOG_EXTRACTS[@]} -gt 0 ]; then
        echo "  Extractions performed:"
        for i in "${LOG_EXTRACTS[@]}"; do echo "    - $i"; done
    else
        echo "  No extractions performed."
    fi

    if [ ${#LOG_FAIL_DOWNLOADS[@]} -gt 0 ] || [ ${#LOG_FAIL_EXTRACTS[@]} -gt 0 ]; then
        echo "  Failures:"
        for i in "${LOG_FAIL_DOWNLOADS[@]}"; do echo "    - download failed: $i"; done
        for i in "${LOG_FAIL_EXTRACTS[@]}"; do echo "    - extract failed: $i"; done
    fi

    mapfile -t AFTER_DIRS < <(find datasets -type d 2>/dev/null || true)

    local already=()
    for d in "${BEFORE_DIRS[@]}"; do
        for ad in "${AFTER_DIRS[@]}"; do
            if [ "$d" = "$ad" ]; then already+=("$d"); break; fi
        done
    done
    if [ ${#already[@]} -gt 0 ]; then
        echo "  Previously existing directories (skipped):"
        for d in "${already[@]}"; do echo "    - $d"; done
    fi

    local new=()
    for ad in "${AFTER_DIRS[@]}"; do
        local found=false
        for d in "${BEFORE_DIRS[@]}"; do
            if [ "$ad" = "$d" ]; then found=true; break; fi
        done
        if [ "$found" = false ]; then new+=("$ad"); fi
    done
    if [ ${#new[@]} -gt 0 ]; then
        echo "  New directories created:"
        for d in "${new[@]}"; do echo "    - $d"; done
    fi
}

trap print_summary EXIT

echo "Downloading datasets. This may take a while..."

if [ ! -d datasets/deblurring/defocus/test/DPDD ]; then
    echo "Downloading DPDD dataset..."
    wget 'https://drive.usercontent.google.com/download?id=1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw&export=download&confirm=t' -O test.zip
    echo "Extracting DPDD dataset..."
    unzip -qd datasets/deblurring/defocus test.zip
    rm test.zip
    mkdir datasets/deblurring/defocus/test/DPDD
    find datasets/deblurring/defocus/test -mindepth 1 -maxdepth 1 ! -name DPDD -exec mv {} datasets/deblurring/defocus/test/DPDD \;
fi

if [ ! -d datasets/deblurring/defocus/test/GoPro ]; then
    echo "Downloading GoPro dataset..."
    wget 'https://drive.usercontent.google.com/download?id=1k6DTSHu4saUgrGTYkkZXTptILyG9RRll&export=download&confirm=t' -O test.zip
    echo "Extracting GoPro dataset..."
    unzip -qd datasets/deblurring/motion test.zip
    rm test.zip
fi

if [ ! -d datasets/deblurring/motion/test/HIDE ]; then
    echo "Downloading HIDE dataset..."
    wget 'https://drive.usercontent.google.com/download?id=1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A&export=download&confirm=t' -O test.zip
    echo "Extracting HIDE dataset..."
    unzip -qd datasets/deblurring/motion test.zip
    rm test.zip
fi

if [ ! -d datasets/deblurring/motion/test/RealBlur-J ]; then
    echo "Downloading RealBlur-J dataset..."
    wget 'https://drive.usercontent.google.com/download?id=1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS&export=download&confirm=t' -O test.zip
    echo "Extracting RealBlur-J dataset..."
    unzip -qd datasets/deblurring/motion test.zip
    rm test.zip
fi

if [ ! -d datasets/deblurring/motion/test/RealBlur-R ]; then
    echo "Downloading RealBlur-R dataset..."
    wget 'https://drive.usercontent.google.com/download?id=1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW&export=download&confirm=t' -O test.zip
    echo "Extracting RealBlur-R dataset..."
    unzip -qd datasets/deblurring/motion test.zip
    rm test.zip
fi

if [ ! -d datasets/denoising/gaussian/test ]; then
    echo "Downloading Gaussian Denoising datasets..."
    wget 'https://drive.usercontent.google.com/download?id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0&export=download&confirm=t' -O test.zip
    echo "Extracting Gaussian Denoising datasets..."
    unzip -qd datasets/denoising/gaussian test.zip
    rm test.zip
fi

if [ ! -d datasets/denoising/real/test/SIDD ]; then
    echo "Downloading SIDD dataset..."
    wget 'https://drive.usercontent.google.com/download?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ&export=download&confirm=t' -O test.zip
    echo "Extracting SIDD dataset..."
    unzip -qd datasets/denoising/real test.zip
    rm test.zip
fi

echo "Finished downloading datasets."
