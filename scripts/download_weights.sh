#!/bin/bash
# Logging helpers: record downloads, extractions and pre-existing dirs; print a summary on exit.
LOG_DOWNLOADS=()
LOG_EXTRACTS=()
LOG_FAIL_DOWNLOADS=()
LOG_FAIL_EXTRACTS=()
LOG_SKIPPED_DOWNLOADS=()
mapfile -t BEFORE_DIRS < <(find weights -type d 2>/dev/null || true)

wget() {
    local args=("$@") out="" destdir="" url="" filename dest
    # parse args to find -O, -P and the URL (first non-option)
    for ((i=0;i<${#args[@]};i++)); do
        case "${args[i]}" in
            -O) out="${args[i+1]}"; i=$((i+1));;
            -P) destdir="${args[i+1]}"; i=$((i+1));;
            --) ;;
            -*) ;;
            *) if [ -z "$url" ]; then url="${args[i]}"; fi;;
        esac
    done

    # if -O specified, that's our destination file
    if [ -n "$out" ]; then
        dest="$out"
    else
        # derive filename from URL (strip query string)
        filename="${url##*/}"
        filename="${filename%%\?*}"
        if [ -n "$destdir" ]; then
            dest="${destdir%/}/$filename"
        else
            dest="$filename"
        fi
    fi

    # if destination exists, skip download
    if [ -f "$dest" ]; then
        echo " [log] exists $dest -> skipping download"
        LOG_SKIPPED_DOWNLOADS+=("$dest")
        return 0
    fi

    echo " [log] downloading ${url} -> ${out:-$dest}"
    if command wget "$@"; then
        LOG_DOWNLOADS+=("${out:-$dest}")
    else
        LOG_FAIL_DOWNLOADS+=("${out:-$dest}")
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

    if [ ${#LOG_SKIPPED_DOWNLOADS[@]} -gt 0 ]; then
        echo "  Skipped downloads (already present):"
        for i in "${LOG_SKIPPED_DOWNLOADS[@]}"; do echo "    - $i"; done
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

    mapfile -t AFTER_DIRS < <(find weights -type d 2>/dev/null || true)

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

echo "Downloading pre-trained weights..."

echo "Downloading REDNet weights..."
mkdir -p weights/REDNet
if [ ! -f weights/REDNet/50.pt ]; then
    wget https://bitbucket.org/chhshen/image-denoising/raw/master/model/denoising/50.caffemodel
    python caffemodel2pytorch/caffemodel2pytorch.py 50.caffemodel -o weights/REDNet/50.pt
    rm -f 50.caffemodel
else
    echo " [log] exists weights/REDNet/50.pt -> skipping download"
    LOG_SKIPPED_DOWNLOADS+=("weights/REDNet/50.pt")
fi

echo "Downloading DnCNN weights..."
mkdir -p weights/DnCNN
for noise in 15 25 50; do
    wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_${noise}.pth -P weights/DnCNN/
done
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth -P weights/DnCNN/
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth -P weights/DnCNN/

echo "Downloading DeblurGANv2 weights..."
mkdir -p weights/DeblurGANv2
wget 'https://drive.usercontent.google.com/download?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&export=download&confirm=t' \
    -O weights/DeblurGANv2/fpn_inception.h5
wget 'https://drive.usercontent.google.com/download?id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU&export=download&confirm=t' \
    -O weights/DeblurGANv2/fpn_mobilenet.h5

echo "Downloading Restormer weights..."
mkdir -p weights/Restormer
mkdir -p weights/Restormer/denoising
wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth \
    -P weights/Restormer/denoising/
wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_blind.pth \
    -P weights/Restormer/denoising/
for noise in 15 25 50; do
    wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma${noise}.pth \
        -P weights/Restormer/denoising/
    wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_sigma${noise}.pth \
        -P weights/Restormer/denoising/
done
wget https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth \
    -P weights/Restormer/denoising/
mkdir -p weights/Restormer/deblurring
wget https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth \
    -P weights/Restormer/deblurring/
wget https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth \
    -P weights/Restormer/deblurring/
wget https://github.com/swz30/Restormer/releases/download/v1.0/dual_pixel_defocus_deblurring.pth \
    -P weights/Restormer/deblurring/

echo "Downloading MaIR weights..."
mkdir -p weights/MaIR
mkdir -p weights/MaIR/denoising
wget 'https://drive.usercontent.google.com/download?id=1XUDCSK1Cs492mopqQrDVLNCC2stO1paA&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s15.pth
wget 'https://drive.usercontent.google.com/download?id=1jIDSzksBracVnyiVSkwFNEX--JOP1H1i&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s25.pth
wget 'https://drive.usercontent.google.com/download?id=1YdhrrPfEZ70JVuJgFdTmSLtFuu2giFdb&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s50.pth
wget 'https://drive.usercontent.google.com/download?id=1M8pDYp_-Yl46pMFqv_tnImJ8w1z6h7bH&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_RealDN.pth
mkdir -p weights/MaIR/deblurring
wget 'https://drive.usercontent.google.com/download?id=1bdYWJ0FXYknQuJQg77KrwII2jJHlX-3k&export=download&confirm=t' \
    -O weights/MaIR/deblurring/MaIR_MotionDeblur.pth

echo "Finished downloading weights."
