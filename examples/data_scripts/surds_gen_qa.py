"""Generate canonical QA pairs from SURDS metadata.

Port of Drive-MLLM/hfdata_to_eval_vqa.py that:
  - reads local metadata.jsonl (no HF re-download, no image re-save)
  - emits one JSONL per split in the canonical schema from
    .claude/autonomy_dataset_curation_plan.md (Phase 1)
  - processes both train and validation
  - keeps the paper's multi-obj/single-obj balancing (single-obj samples
    are subsampled to match the multi-obj VQA count)

Schema per line:
  sample_id, image_path, task_family, template_type,
  prompt, answer, options, metadata
"""

import argparse
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TASK_FAMILY = 'spatial_relation_geometry'

TEMPLATE_FILES = {
    'yaw':      '00_yaw_vqa.txt',
    'xy2d':     '01_xy2d_vqa.txt',
    'depth':    '02_depth_vqa.txt',
    'distance': '03_dis_vqa.txt',
    'lr':       '04_lf_vqa.txt',
    'fb':       '05_fb_vqa.txt',
}

OPPOSITE_YAW = {
    'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East',
    'Northeast': 'Southwest', 'Southeast': 'Northwest',
    'Southwest': 'Northeast', 'Northwest': 'Southeast',
}


def load_templates(prompt_dir: Path) -> dict[str, str]:
    return {k: (prompt_dir / fn).read_text() for k, fn in TEMPLATE_FILES.items()}


def format_range(r: list[int]) -> str:
    a, b = r
    return f"Between {a} {'meter' if a <= 1 else 'meters'} and {b} {'meter' if b <= 1 else 'meters'}"


def generate_depth_range(depth: float) -> tuple[list[int], list[int], list[int]]:
    if depth < 7:
        L = random.uniform(6, 7)
        ans = [max(1, round(depth - L / 2)), round(depth + L / 2)]
        r2s = ans[1] + random.randint(1, 2); r2e = r2s + random.randint(3, 4)
        r3s = r2e + random.randint(1, 2);    r3e = r3s + random.randint(3, 4)
        return ans, [r2s, r2e], [r3s, r3e]
    if depth > 15:
        L = random.uniform(6, 7)
        ans = [round(depth - L / 2), round(depth + L / 2)]
        r2e = ans[0] - random.randint(1, 2); r2s = r2e - random.randint(3, 4)
        r3e = r2s - random.randint(1, 2);    r3s = max(1, r3e - random.randint(3, 4))
        return ans, [r2s, r2e], [r3s, r3e]
    L = random.uniform(6, 7)
    ans = [round(depth - L / 2), round(depth + L / 2)]
    r2e = ans[0] - random.randint(1, 2); r2s = max(1, r2e - random.randint(3, 4))
    r3s = ans[1] + random.randint(1, 2); r3e = r3s + random.randint(3, 4)
    return ans, [r2s, r2e], [r3s, r3e]


def load_metadata(jsonl: Path) -> list[dict]:
    with jsonl.open() as f:
        return [json.loads(line) for line in f]


def emit(qas: list, split: str, tmpl: str, idx: int, image_path: str,
         prompt: str, answer: str, options, extra: dict):
    qas.append({
        'sample_id': f'surds_{split}_{idx:05d}_{tmpl}',
        'image_path': image_path,
        'task_family': TASK_FAMILY,
        'template_type': tmpl,
        'prompt': prompt,
        'answer': answer,
        'options': options,
        'metadata': extra,
    })


def process_split(split: str, meta_path: Path, image_root: Path,
                  templates: dict[str, str]) -> list[dict]:
    meta = load_metadata(meta_path)
    qas: list[dict] = []
    multi_idx: list[int] = []

    # Pass 1 — multi-object samples.
    for i, d in enumerate(tqdm(meta, desc=f'{split} multi-obj')):
        descs = d['descs']
        if len(descs) <= 1:
            continue
        multi_idx.append(i)
        img_path = str(image_root / d['file_name'])
        dists, xy2ds, depths = d['distances'], d['xy2Ds'], d['depths']

        for a in range(len(descs) - 1):
            for b in range(a + 1, len(descs)):
                o1, o2 = 'the ' + descs[a], 'the ' + descs[b]
                O1, O2 = o1.capitalize(), o2.capitalize()

                # distance (closer / farther)
                if abs(dists[a] - dists[b]) <= 1:
                    c_ans = f_ans = 'Almost the same'
                elif dists[a] < dists[b]:
                    c_ans, f_ans = O1, O2
                else:
                    c_ans, f_ans = O2, O1
                opts = [O1, O2, 'Almost the same']
                for direction, ans in (('closer', c_ans), ('farther', f_ans)):
                    emit(qas, split, 'distance', len(qas), img_path,
                         templates['distance'].format(o1, o2, direction, O1, O2),
                         ans, opts,
                         {'obj1': descs[a], 'obj2': descs[b],
                          'dist1': dists[a], 'dist2': dists[b],
                          'direction': direction})

                # left / right
                x1, x2 = xy2ds[a][0], xy2ds[b][0]
                if abs(x1 - x2) < 100:
                    l_ans = r_ans = 'Almost the same'
                elif x1 < x2:
                    l_ans, r_ans = O1, O2
                else:
                    l_ans, r_ans = O2, O1
                opts = [O1, O2, 'Almost the same']
                for side, ans in (('left', l_ans), ('right', r_ans)):
                    emit(qas, split, 'lr', len(qas), img_path,
                         templates['lr'].format(side, o1, o2, O1, O2),
                         ans, opts,
                         {'obj1': descs[a], 'obj2': descs[b],
                          'x1': x1, 'x2': x2, 'side': side})

                # front / back
                d1, d2 = depths[a], depths[b]
                if abs(d1 - d2) < 0.5:
                    f_ans = b_ans = 'Almost the same in terms of front-back position'
                elif d1 > d2:
                    f_ans, b_ans = 'Yes', 'No'
                else:
                    f_ans, b_ans = 'No', 'Yes'
                opts = ['Yes', 'No', 'Almost the same in terms of front-back position']
                for relation, ans in (('in front of', f_ans), ('behind', b_ans)):
                    emit(qas, split, 'fb', len(qas), img_path,
                         templates['fb'].format(o1, relation, o2),
                         ans, opts,
                         {'obj1': descs[a], 'obj2': descs[b],
                          'depth1': d1, 'depth2': d2, 'relation': relation})

    multi_vqa_n = sum(1 for q in qas if q['template_type'] == 'distance') // 2
    logger.info(f'[{split}] multi-obj samples: {len(multi_idx)}, '
                f'multi-obj VQAs per template: {multi_vqa_n}')

    # Pass 2 — single-object samples (subsampled to multi_vqa_n).
    single_idx = [i for i in range(len(meta)) if i not in set(multi_idx)]
    chosen = sorted(random.sample(single_idx, min(multi_vqa_n, len(single_idx))))

    for i in tqdm(chosen, desc=f'{split} single-obj'):
        d = meta[i]
        img_path = str(image_root / d['file_name'])
        obj = 'the ' + d['descs'][0]
        bbox = d['bboxes2D'][0]

        # yaw
        yaw_desc = d['yaw_descs'][0]
        if yaw_desc in {'Northeast', 'Southeast', 'Northwest', 'Southwest'}:
            opts = random.sample(['Northeast', 'Southeast', 'Northwest', 'Southwest'], k=4)
        elif yaw_desc in {'East', 'South', 'West', 'North'}:
            opts = random.sample(['East', 'South', 'West', 'North'], k=4)
        else:
            raise ValueError(f'unknown yaw: {yaw_desc}')
        for cam, ans in (('North', yaw_desc), ('South', OPPOSITE_YAW[yaw_desc])):
            emit(qas, split, 'yaw', len(qas), img_path,
                 templates['yaw'].format(cam, obj, *opts),
                 ans, opts,
                 {'obj': d['descs'][0], 'obj_bbox': bbox, 'camera': cam,
                  'raw_yaw': d['yaws'][0], 'raw_yaw_desc': yaw_desc})

        # xy2d
        xy = d['xy2Ds'][0]
        emit(qas, split, 'xy2d', len(qas), img_path,
             templates['xy2d'].format(obj), str(xy), None,
             {'obj': d['descs'][0], 'obj_bbox': bbox, 'xy2d': xy})

        # depth
        ans_r, r2, r3 = generate_depth_range(d['depths'][0])
        opts = random.sample([format_range(ans_r), format_range(r2), format_range(r3)], k=3)
        emit(qas, split, 'depth', len(qas), img_path,
             templates['depth'].format(obj, *opts),
             format_range(ans_r), opts,
             {'obj': d['descs'][0], 'obj_bbox': bbox, 'raw_depth': d['depths'][0]})

    return qas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--surds_root', default='/mnt/data4/shasta/amar.amarjyoti/research_data/raw/surds')
    p.add_argument('--prompt_dir', default='/mnt/data4/shasta/amar.amarjyoti/research_data/Drive-MLLM/prompt/prompts_reasoning')
    p.add_argument('--out_dir', default='/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds')
    p.add_argument('--splits', nargs='+', default=['train', 'validation'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Optional cap per split (for dry-runs).')
    args = p.parse_args()

    random.seed(args.seed)
    templates = load_templates(Path(args.prompt_dir))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = Path(args.surds_root) / split
        meta_path = split_dir / 'metadata.jsonl'

        # Honor max_samples by temporarily truncating the stream.
        if args.max_samples is not None:
            tmp_meta = load_metadata(meta_path)[: args.max_samples]
            meta_path = out_dir / f'.{split}_truncated.jsonl'
            with meta_path.open('w') as f:
                for row in tmp_meta:
                    f.write(json.dumps(row) + '\n')

        qas = process_split(split, meta_path, split_dir, templates)
        out_path = out_dir / f'{split}_qa.jsonl'
        with out_path.open('w') as f:
            for q in qas:
                f.write(json.dumps(q) + '\n')
        logger.info(f'[{split}] wrote {len(qas):,} QAs -> {out_path}')

        # per-template counts
        counts: dict[str, int] = {}
        for q in qas:
            counts[q['template_type']] = counts.get(q['template_type'], 0) + 1
        logger.info(f'[{split}] counts: {counts}')

        if args.max_samples is not None:
            meta_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
