#!/bin/bash
# Render Dance and Uneven Floor in every style into one downloadable folder.
# Safe to rerun: finished trainings, recordings and videos are skipped.
# Usage (from the repo root): bash headless/render_batch.sh [parallel_renders, default 3]
export CUDA_VISIBLE_DEVICES=0
JOBS=${1:-3}
STYLES="angry anim_keyframed clown clownangry clownangryOne happy robot sad scared zombiescared"
CLIPS="Dance_Neutral_AverageJoe Uneven_Floor_Neutral_AverageJoe"
OUT=render_results/dance_unevenfloor
LOGS=render_results/logs
mkdir -p $OUT $LOGS
RUN="conda run --no-capture-output -n stymo"
export PMS=$($RUN python -c "import pymotion.render.blender as b;print(b.pymotion_blender_script)")
export OUT LOGS RUN

for style in $STYLES; do
    R=motion/mocha/$style/results/Transformer-py312
    if [ ! -f $R/model.pt ]; then
        echo "[$(date +%H:%M)] train $style"
        $RUN python src/stylizer_train.py motion/mocha/$style/ motion/mocha/neutral/ --epochs 100 --name py312 \
            > $LOGS/train_$style.log 2>&1 || { echo "FAILED train $style"; continue; }
    fi
    for clip in $CLIPS; do
        [ -f $R/input/$clip.bvh ] || cp motion/mocha/neutral/$clip.bvh $R/input/  # models trained on fewer inputs
        if [ ! -f $R/headless/$clip/commands.json ]; then
            echo "[$(date +%H:%M)] stylize $style $clip"
            $RUN python headless/stymo_headless.py $R --clips $clip > $LOGS/stylize_${style}_$clip.log 2>&1 \
                || echo "FAILED stylize $style $clip"
        fi
    done
done

render_one() {
    style=$1; clip=$2
    D=motion/mocha/$style/results/Transformer-py312/headless/$clip
    dest=$OUT/${clip%%_Neutral_AverageJoe}__$style.mp4
    [ -f $D/commands.json ] || { echo "SKIPPED $style $clip (no recording)"; return; }
    if [ ! -f $D/video.mp4 ]; then
        echo "[$(date +%H:%M)] render $style $clip"
        $RUN blender -b --factory-startup -P headless/blender_replay.py -- $D --pymotion_script $PMS \
            --blend $D/scene.blend --mp4 $D/video.mp4.tmp > $LOGS/render_${style}_$clip.log 2>&1 \
            && mv $D/video.mp4.tmp $D/video.mp4 || { echo "FAILED render $style $clip"; return; }
    fi
    cp $D/video.mp4 "$dest" && echo "[$(date +%H:%M)] done $dest"
}
export -f render_one

for style in $STYLES; do for clip in $CLIPS; do echo "$style $clip"; done; done \
    | xargs -P "$JOBS" -L 1 bash -c 'render_one "$0" "$1"'

echo "Videos in $OUT: $(ls $OUT/*.mp4 2>/dev/null | wc -l) of 20"
