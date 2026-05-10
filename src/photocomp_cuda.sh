#!/bin/bash
# ============================================================
# SGM CUDA Benchmark Script
# ============================================================

EXEC="./sgmCuda"
N_RUNS=20
IMAGES=("bull" "cones" "teddy" "venus")
SUFFIX="cuda"
RESULTS_FILE="results_cuda.txt"

# ============================================================

if [ ! -x "$EXEC" ]; then
    echo "ERROR: Executable '$EXEC' not found or not executable." >&2
    exit 1
fi

exec > >(tee "$RESULTS_FILE")

compute_stats() {
    printf '%s\n' "$@" | awk '
    BEGIN { n=0; sum=0; min=1e18; max=-1e18 }
    { n++; sum+=$1; a[n]=$1; if($1<min)min=$1; if($1>max)max=$1 }
    END {
        if (n==0) { printf "0.00 0.00 0.00 0.00\n"; exit }
        mean = sum/n; var = 0
        for (i=1; i<=n; i++) var += (a[i]-mean)^2
        std = (n>1) ? sqrt(var/(n-1)) : 0
        printf "%.2f %.2f %.2f %.2f\n", mean, std, min, max
    }'
}

declare -a SUM_IMG SUM_CORRECT SUM_H_MEAN SUM_D_MEAN SUM_SPEEDUP

echo ""
echo "======================================================================"
echo "  SGM CUDA Benchmark"
printf "  Runs per image : %d  (1 warmup + %d timed)\n" "$N_RUNS" "$((N_RUNS-1))"
echo "  Date           : $(date)"
echo "======================================================================"

for img in "${IMAGES[@]}"; do
    img_left="l${img}.pgm"
    img_right="r${img}.pgm"
    out_host="h_d${img}_${SUFFIX}.pgm"
    out_device="d_d${img}_${SUFFIX}.pgm"

    echo ""
    echo "  ------------------------------------------------------------------"
    echo "  Processing image: $img"
    echo "  ------------------------------------------------------------------"

    # ---- Correctness check (one dedicated run before timing) ----
    "$EXEC" -l "$img_left" -r "$img_right" -t "$out_host" -o "$out_device" > /dev/null 2>&1

    correctness="FAIL"
    if [ -f "$out_host" ] && [ -f "$out_device" ]; then
        if ./testDiffs "$out_host" "$out_device" > /dev/null 2>&1; then
            correctness="PASS"
        else
            echo "  WARNING: testDiffs reported differences for '$img'." >&2
        fi
    else
        echo "  WARNING: Output files missing after correctness run for '$img'." >&2
    fi

    # ---- Timing runs ----
    host_times=()
    dev_times=()

    for ((run=1; run<=N_RUNS; run++)); do
        output=$("$EXEC" -l "$img_left" -r "$img_right" -t "$out_host" -o "$out_device" 2>&1)

        # Discard first run (GPU warmup / cache effects)
        [ "$run" -eq 1 ] && continue

        host_val=$(echo "$output" | grep -i "Host processing time:" | grep -oE '[0-9]+([.][0-9]+)?' | head -1)
        dev_val=$(echo "$output"  | grep -i "Device processing time:" | grep -oE '[0-9]+([.][0-9]+)?' | head -1)

        if [ -z "$host_val" ] || [ -z "$dev_val" ]; then
            echo "  WARNING: Could not parse timing from run $run for '$img' — skipping." >&2
            continue
        fi

        host_times+=("$host_val")
        dev_times+=("$dev_val")
    done

    n_timed=${#host_times[@]}

    if [ "$n_timed" -eq 0 ]; then
        echo "  ERROR: No valid timing data collected for '$img'." >&2
        SUM_IMG+=("$img")
        SUM_CORRECT+=("$correctness")
        SUM_H_MEAN+=("N/A")
        SUM_D_MEAN+=("N/A")
        SUM_SPEEDUP+=("N/A")
        continue
    fi

    # ---- Statistics ----
    read -r h_mean h_std h_min h_max <<< "$(compute_stats "${host_times[@]}")"
    read -r d_mean d_std d_min d_max <<< "$(compute_stats "${dev_times[@]}")"
    speedup=$(awk "BEGIN { printf \"%.1f\", $h_mean / $d_mean }")

    # ---- Per-image results table ----
    echo ""
    echo "  Image: $img"
    echo "  Correctness: $correctness"
    printf "  Runs: %d (1 warmup discarded)\n" "$n_timed"
    echo ""
    printf "  %-12s %14s %14s %12s\n" ""         "Host (ms)"  "Device (ms)" "Speedup"
    printf "  %-12s %14s %14s %12s\n" "Mean:"    "$h_mean"    "$d_mean"     "${speedup}x"
    printf "  %-12s %14s %14s\n"      "Std Dev:" "$h_std"     "$d_std"
    printf "  %-12s %14s %14s\n"      "Min:"     "$h_min"     "$d_min"
    printf "  %-12s %14s %14s\n"      "Max:"     "$h_max"     "$d_max"
    echo ""

    SUM_IMG+=("$img")
    SUM_CORRECT+=("$correctness")
    SUM_H_MEAN+=("$h_mean")
    SUM_D_MEAN+=("$d_mean")
    SUM_SPEEDUP+=("$speedup")
done

# ---- Final summary table ----
echo ""
echo "======================================================================"
echo "  SUMMARY"
echo "======================================================================"
printf "  %-8s  %-12s  %14s  %14s  %10s\n" \
    "Image" "Correctness" "Mean Host(ms)" "Mean Dev(ms)" "Speedup"
echo "  ------------------------------------------------------------------"
for ((i=0; i<${#SUM_IMG[@]}; i++)); do
    if [ "${SUM_H_MEAN[$i]}" = "N/A" ]; then
        printf "  %-8s  %-12s  %14s  %14s  %10s\n" \
            "${SUM_IMG[$i]}" "${SUM_CORRECT[$i]}" "N/A" "N/A" "N/A"
    else
        printf "  %-8s  %-12s  %14s  %14s  %10s\n" \
            "${SUM_IMG[$i]}" "${SUM_CORRECT[$i]}" \
            "${SUM_H_MEAN[$i]}" "${SUM_D_MEAN[$i]}" "${SUM_SPEEDUP[$i]}x"
    fi
done
echo ""
echo "  Full results saved to: $RESULTS_FILE"
echo ""
