import os
import gc
import time
import uuid
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from model import *
from utils import *
from impact import (
    append_live_progress,
    apply_effective_pxp_impact,
    build_impact_config,
    ensure_impact_ppp_diag,
    ensure_impact_pxp_diag,
    format_result_line,
    impact_active_for_scope,
    normalize_method,
    propagation_cache_path,
    run_debug_impact_toy_tests,
)


def load_diag_tensor(args, diag_name):
    candidates = [
        Path(args.emb_path) / diag_name,
        Path(diag_name),
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path)
    raise FileNotFoundError(
        f"Could not find {diag_name}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_label_self_effect_diag(g, args, impact_config, key):
    if impact_active_for_scope(impact_config, 'label'):
        if key == 'PPP':
            return torch.load(ensure_impact_ppp_diag(g, args, impact_config))
        if key == 'PAP':
            return torch.load(ensure_impact_pxp_diag(g, args, impact_config, 'A', 'PAP'))
        if key == 'PFP':
            return torch.load(ensure_impact_pxp_diag(g, args, impact_config, 'F', 'PFP'))
    return load_diag_tensor(args, f'{args.dataset}_{key}_diag.pt')


def remove_label_self_effect(label_feat, label_onehot, diag, key):
    corrected = label_feat - diag.unsqueeze(-1) * label_onehot
    min_val = float(corrected.min().item())
    if min_val < -1e-5:
        neg_count = int((corrected < -1e-5).sum().item())
        raise RuntimeError(
            f"Self-effect removal for {key} produced {neg_count} negative entries "
            f"(min={min_val:.6g}); the diagonal does not match the propagation operator."
        )
    return corrected.clamp_min_(0)


def parse_impact_stages(spec, num_stages):
    spec = str(spec).strip().lower()
    if spec in ("all", "*"):
        return set(range(num_stages))
    if spec in ("none", ""):
        return set()

    stages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError(f"Invalid --impact-stages range: {part}")
            stages.update(range(start, end + 1))
        else:
            stages.add(int(part))

    invalid = [stage for stage in stages if stage < 0 or stage >= num_stages]
    if invalid:
        raise ValueError(
            f"--impact-stages contains invalid stages {invalid}; "
            f"valid range is 0..{num_stages - 1}"
        )
    return stages


def impact_enabled_for_stage(impact_stages, stage):
    return stage in impact_stages


def args_with_impact_method(args, method):
    proxy = argparse.Namespace(**vars(args))
    proxy.impact_method = method
    if method == "none":
        proxy.impact_apply_to = "both"
    return proxy


def snapshot_node_data(g):
    return {
        ntype: {key: value for key, value in g.nodes[ntype].data.items()}
        for ntype in g.ntypes
    }


def restore_node_data(g, saved):
    for ntype in g.ntypes:
        for key in list(g.nodes[ntype].data.keys()):
            g.nodes[ntype].data.pop(key)
        for key, value in saved.get(ntype, {}).items():
            g.nodes[ntype].data[key] = value


def prepare_mag_feature_tensors(
    args,
    g,
    tgt_type,
    impact_config,
    feature_impact_active,
    extra_metapath,
    max_hops,
    init2sort,
    initial_node_data,
):
    cache_args = args if feature_impact_active else args_with_impact_method(args, "none")
    active_config = impact_config if feature_impact_active else None
    cache_path = propagation_cache_path(cache_args, 'feature', args.num_hops, max_hops, extra_metapath)
    variant = "impact" if feature_impact_active else "baseline"

    if cache_path is not None and cache_path.exists():
        print(f'Load cached {variant} feature propagation from {cache_path}', flush=True)
        feats = torch.load(cache_path, map_location='cpu')
    else:
        restore_node_data(g, initial_node_data)
        g = hg_propagate(
            g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False,
            impact_config=active_config, impact_scope='feature')
        apply_effective_pxp_impact(g, active_config, 'feature', source_key='P')

        feats = {}
        keys = list(g.nodes[tgt_type].data.keys())
        print(f'Involved {variant} feat keys {keys}')
        for k in keys:
            feats[k] = g.nodes[tgt_type].data.pop(k)

        g = clear_hg(g, echo=False)
        if cache_path is not None:
            torch.save(feats, cache_path)
            print(f'Saved {variant} feature propagation cache to {cache_path}', flush=True)

    return {k: v[init2sort] for k, v in feats.items()}


def main(args):
    if args.seed >= 0:
        set_random_seed(args.seed)
        
    g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset(args)
    args.impact_method = normalize_method(args.impact_method)
    impact_config = build_impact_config(args, getattr(g, '_impact_paper_year', None))
    impact_stages = parse_impact_stages(args.impact_stages, len(args.stages))
    print(
        f"Impact method={impact_config.method}, apply_to={impact_config.apply_to}, "
        f"gsmp_first_layer_only={impact_config.gsmp_first_layer_only}, "
        f"impact_stages={sorted(impact_stages)}",
        flush=True,
    )
    run_tic = time.time()

    # =======
    # rearange node idx (for feats & labels)
    # =======
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

    if total_num_nodes < num_nodes:
        flag = torch.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = torch.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = torch.tensor([], dtype=torch.long)

    init2sort = torch.cat([train_nid, val_nid, test_nid, extra_nid])
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    # =======
    # features propagate alongside the metapath
    # =======
    prop_tic = datetime.datetime.now()

    if args.dataset == 'ogbn-mag': # multi-node-types & multi-edge-types
        tgt_type = 'P'
        initial_node_data = snapshot_node_data(g)

        extra_metapath = [] # ['AIAP', 'PAIAP']
        extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

        print(f'Current num hops = {args.num_hops}')
        if len(extra_metapath):
            max_hops = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_hops = args.num_hops + 1

        active_feature_stages = [
            stage for stage in range(args.start_stage, len(args.stages))
            if impact_enabled_for_stage(impact_stages, stage)
            and impact_active_for_scope(impact_config, 'feature')
        ]
        inactive_feature_stages = [
            stage for stage in range(args.start_stage, len(args.stages))
            if not (
                impact_enabled_for_stage(impact_stages, stage)
                and impact_active_for_scope(impact_config, 'feature')
            )
        ]
        feats_by_impact = {}
        if active_feature_stages:
            feats_by_impact[True] = prepare_mag_feature_tensors(
                args, g, tgt_type, impact_config, True, extra_metapath,
                max_hops, init2sort, initial_node_data)
        if inactive_feature_stages or not active_feature_stages:
            feats_by_impact[False] = prepare_mag_feature_tensors(
                args, g, tgt_type, impact_config, False, extra_metapath,
                max_hops, init2sort, initial_node_data)
        g = clear_hg(g, echo=False)
    else:
        assert 0

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # train_loader = torch.utils.data.DataLoader(
    #     torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
    # eval_loader = full_loader = []
    all_loader = torch.utils.data.DataLoader(
        torch.arange(num_nodes), batch_size=args.batch_size, shuffle=False, drop_last=False)

    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
    labels_cuda = labels.long().to(device)

    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print(checkpt_file)

    for stage in range(args.start_stage, len(args.stages)):
        epochs = args.stages[stage]
        stage_impact_active = impact_enabled_for_stage(impact_stages, stage)
        feature_impact_active = stage_impact_active and impact_active_for_scope(impact_config, 'feature')
        label_impact_config = (
            impact_config
            if stage_impact_active and impact_active_for_scope(impact_config, 'label')
            else None
        )
        feats = feats_by_impact[feature_impact_active]
        print(
            f"Stage {stage} impact: feature={feature_impact_active}, "
            f"label={label_impact_config is not None}",
            flush=True,
        )

        if len(args.reload):
            pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
            assert os.path.exists(pt_path)
            print(f'Reload raw_preds from {pt_path}', flush=True)
            raw_preds = torch.load(pt_path, map_location='cpu')

        # =======
        # Expand training set & train loader
        # =======
        if stage > 0:
            preds = raw_preds.argmax(dim=-1)
            predict_prob = raw_preds.softmax(dim=1)

            train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
            test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

            print(f'Stage {stage-1} history model:\n\t' \
                + f'Train acc {train_acc*100:.4f} Val acc {val_acc*100:.4f} Test acc {test_acc*100:.4f}')

            confident_mask = predict_prob.max(1)[0] > args.threshold
            val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
            test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
            val_enhance_nid     = val_enhance_offset + trainval_point
            test_enhance_nid    = test_enhance_offset + valtest_point
            enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

            print(f'Stage: {stage}, threshold {args.threshold}, confident nodes: {len(enhance_nid)} / {total_num_nodes - trainval_point}')
            val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
            print(f'\t\t val confident nodes: {len(val_enhance_nid)} / {valid_node_nums},  val confident level: {val_confident_level}')
            test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
            print(f'\t\ttest confident nodes: {len(test_enhance_nid)} / {test_node_nums}, test confident_level: {test_confident_level}')

            del train_loader
            train_batch_size = int(args.batch_size * len(train_nid) / (len(enhance_nid) + len(train_nid)))
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=train_batch_size, shuffle=True, drop_last=False)
            enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_nid)))
            enhance_loader = torch.utils.data.DataLoader(
                enhance_nid, batch_size=enhance_batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if stage > 0:
                label_onehot = predict_prob[sort2init].clone()
            else:
                label_onehot = torch.zeros((num_nodes, n_classes))
            label_onehot[train_nid] = F.one_hot(init_labels[train_nid], n_classes).float()

            if args.dataset == 'ogbn-mag':
                g.nodes['P'].data['P'] = label_onehot

                extra_metapath = [] # ['PAIAP']
                extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]

                print(f'Current num label hops = {args.num_label_hops}')
                if len(extra_metapath):
                    max_hops = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
                else:
                    max_hops = args.num_label_hops + 1

                g = hg_propagate(
                    g, tgt_type, args.num_label_hops, max_hops, extra_metapath, echo=False,
                    impact_config=label_impact_config, impact_scope='label')
                apply_effective_pxp_impact(g, label_impact_config, 'label', source_key='P')

                keys = list(g.nodes[tgt_type].data.keys())
                print(f'Involved label keys {keys}')
                for k in keys:
                    if k == tgt_type: continue
                    label_feats[k] = g.nodes[tgt_type].data.pop(k)
                g = clear_hg(g, echo=False)

                # label_feats = remove_self_effect_on_label_feats(label_feats, label_onehot)
                for k in ['PPP', 'PAP', 'PFP', 'PPPP', 'PAPP', 'PPAP', 'PFPP', 'PPFP']:
                    if k in label_feats:
                        diag = load_label_self_effect_diag(g, args, label_impact_config, k)
                        label_feats[k] = remove_label_self_effect(label_feats[k], label_onehot, diag, k)

                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                label_emb = (label_feats['PPP'] + label_feats['PAP'] + label_feats['PP'] + label_feats['PFP']) / 4
                check_acc({'label_emb': label_emb}, condition, init_labels, train_nid, val_nid, test_nid)
        else:
            label_emb = torch.zeros((num_nodes, n_classes))

        label_feats = {k: v[init2sort] for k, v in label_feats.items()}
        label_emb = label_emb[init2sort]

        if stage == 0:
            label_feats = {}

        # =======
        # Eval loader
        # =======
        if stage > 0:
            del eval_loader
        eval_loader = []
        for batch_idx in range((num_nodes-trainval_point-1) // args.batch_size + 1):
            batch_start = batch_idx * args.batch_size + trainval_point
            batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + trainval_point)

            batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
            batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
            batch_labels_emb = label_emb[batch_start:batch_end]
            eval_loader.append((batch_feats, batch_label_feats, batch_labels_emb))

        data_size = {k: v.size(-1) for k, v in feats.items()}

        # =======
        # Construct network
        # =======
        model = MOE(args.dataset, data_size, args.embed_size,
            args.hidden, n_classes,
            len(feats), len(label_feats), tgt_type,
            dropout=args.dropout,
            input_drop=args.input_drop,
            att_drop=args.att_drop,
            label_drop=args.label_drop,
            n_layers_1=args.n_layers_1,
            n_layers_2=args.n_layers_2,
            n_layers_3=args.n_layers_3,
            act=args.act,
            residual=args.residual,
            bns=args.bns, label_bns=args.label_bns,
            label_residual=args.label_residual, 
            num_experts=args.num_experts,
            aggregation=args.aggregation,
            similarity_threshold=args.similarity_threshold,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound
            )
        model = model.to(device)
        if stage == args.start_stage:
            print(model)
            print("# Params:", get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
       
        best_epoch = 0
        best_val_acc = 0
        best_test_acc = 0
        count = 0

        for epoch in range(epochs):
            if args.gc_every > 0 and epoch % args.gc_every == 0:
                gc.collect()
            if args.empty_cache_every_epoch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            start = time.time()
            if stage == 0:
                loss, acc = train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, lamb=args.lamb, scalar=scalar)
            else:
                loss, acc = train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, predict_prob, args.gama, lamb=args.lamb, scalar=scalar)
            end = time.time()

            log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100)
            if 'MOHE' in args.aggregation or "HOPE" in args.aggregation:
                log += "Coverage ratio {:.4f}\n".format(model.mohe.num_covered / model.mohe.num_nodes)
                model.mohe.init_coverage_ratio()
            if args.empty_cache_every_epoch and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if epoch % args.eval_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_preds = []

                    start = time.time()
                    for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
                        batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                        batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                        batch_labels_emb = batch_labels_emb.to(device)
                        x = model(batch_feats, batch_label_feats, batch_labels_emb).cpu()
                        raw_preds.append(x)
                    raw_preds = torch.cat(raw_preds, dim=0)

                    loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
                    loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

                    preds = raw_preds.argmax(dim=-1)
                    val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
                    test_acc = evaluator(preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

                    # 计算moe和shared准确率的并集
                    end = time.time()
                    log += f'Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
                    log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)

                eval_epoch = epoch + 1
                if val_acc > best_val_acc:
                    best_epoch = eval_epoch
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                    torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                    count = 0
                else:
                    count = count + args.eval_every
                    if count >= args.patience:
                        break
                elapsed_sec = time.time() - run_tic
                result_line = format_result_line(
                    args.impact_method, args.seed, stage, eval_epoch, acc,
                    val_acc, test_acc, best_epoch, best_val_acc,
                    best_test_acc, elapsed_sec)
                append_live_progress(
                    args.progress_file, args.impact_method, args.seed, stage,
                    eval_epoch, acc, val_acc, test_acc, best_epoch,
                    best_val_acc, best_test_acc, elapsed_sec)
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}\n{}".format(
                    best_epoch, best_val_acc*100, best_test_acc*100, result_line)
            print(log, flush=True)

        print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))
    
        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        raw_preds = gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)
        torch.save(raw_preds, checkpt_file+f'_{stage}.pt')

    for stage in range(args.start_stage, len(args.stages)):
        file_path = checkpt_file + f'_{stage}.pkl'
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = checkpt_file + f'_{stage}.pt'
        if os.path.exists(file_path):
            os.remove(file_path)
    return [best_val_acc, best_test_acc]


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='HOPE')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1, 2, 3],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default='./dataset/')
    parser.add_argument("--stages", nargs='+',type=int, default=[300, 300, 300, 300],
                        help="The epoch setting for each stage.")
    ## For pre-processing
    parser.add_argument("--emb_path", type=str, default='./dataset/ogbn_mag/')
    parser.add_argument("--extra-embedding", type=str, default='Line',
                        help="the name of extra embeddings")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=True,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=2,
                        help="number of layers of the downstream task")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of layers of residual label connection")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--label-residual", action='store_true', default=False,
                        help="whether to connect the label features")
    parser.add_argument("--act", type=str, default='leaky_relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=True,
                        help="whether to process the input features")
    parser.add_argument("--label-bns", action='store_true', default=True,
                        help="whether to process the input label features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=True,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="the threshold of multi-stage learning, confident nodes "
                           + "whose score above this threshold would be added into the training set")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="compatibility option for the official HOPE reproduction command")
    parser.add_argument("--gama", type=float, default=10,
                        help="parameter for the KL loss")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--reload", type=str, default='')
    parser.add_argument("--aggregation", type=str, default='MOHE',
                        help="the aggregation method for the model")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="the capacity factor for the MoE routing(sigma)")
    parser.add_argument("--lower-bound", type=float, default=0.5,
                        help="the lower capacity factor for the MoE routing")
    parser.add_argument("--upper-bound", type=float, default=3,
                        help="the upper capacity factor for the MoE routing")
    parser.add_argument("--num-experts", type=int, default=10,
                        help="number of HOPE prototype experts")
    parser.add_argument("--lamb", type=float, default=0.5,
                        help="parameter for the prototype separation loss")
    parser.add_argument("--use-sparse-tools", action='store_true', default=True,
                        help="whether to use sparse_tools")
    parser.add_argument("--impact-method", choices=("none", "smp", "gsmp"), default="none",
                        help="temporal impact propagation method")
    parser.add_argument("--impact-apply-to", choices=("feature", "label", "both"), default="both",
                        help="apply SMP/GSMP to feature propagation, label propagation, or both")
    parser.add_argument("--impact-stages", type=str, default="all",
                        help="stages where SMP/GSMP is active: all, none, a single stage like 0, "
                             "or comma/range syntax like 0,2-3")
    parser.add_argument("--impact-gsmp-first-layer-only", action='store_true', default=True,
                        help="apply GSMP only to the first paper-paper propagation step")
    parser.add_argument("--no-impact-gsmp-first-layer-only",
                        dest="impact_gsmp_first_layer_only", action='store_false',
                        help="debug/ablation: apply GSMP to every eligible paper-paper step")
    parser.add_argument("--impact-cache-dir", type=str, default="./impact_cache",
                        help="where to cache propagated feature tensors")
    parser.add_argument("--cache-propagation", action='store_true', default=True,
                        help="cache expensive propagated feature tensors")
    parser.add_argument("--no-cache-propagation", dest="cache_propagation", action='store_false',
                        help="disable propagated feature tensor cache")
    parser.add_argument("--progress-file", type=str, default="./results/live_progress.tsv",
                        help="append-only TSV used for real-time monitoring")
    parser.add_argument("--debug-impact-toy-test", action='store_true', default=False,
                        help="run lightweight SMP/GSMP toy checks and exit")
    parser.add_argument("--gc-every", type=int, default=1,
                        help="run Python garbage collection every N epochs; 0 disables epoch-level gc")
    parser.add_argument("--empty-cache-every-epoch", action='store_true', default=True,
                        help="call torch.cuda.empty_cache around every epoch")
    parser.add_argument("--no-empty-cache-every-epoch",
                        dest="empty_cache_every_epoch", action='store_false',
                        help="skip per-epoch CUDA cache flushing for faster steady-state training")

    return parser.parse_args(args)


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    if args.debug_impact_toy_test:
        run_debug_impact_toy_tests()
        raise SystemExit(0)
    assert args.dataset.startswith('ogbn')
    print(args)

    results = []
    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        results.append(main(args))
    results = np.array(results)
    print('Val:', results[:,0], np.mean(results[:,0]), np.std(results[:,0]))
    print('Test:', results[:,1], np.mean(results[:,1]), np.std(results[:,1]))
    end = time.time()
    print("All Time(s): {:.4f}".format(end - start))
