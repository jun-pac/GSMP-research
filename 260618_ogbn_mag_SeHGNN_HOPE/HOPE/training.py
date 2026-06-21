import os
import gc
import time
import uuid
import argparse
import datetime
import hashlib
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from model import *
from utils import *


def print_resolved_config(args):
    print("Resolved configuration:", flush=True)
    for key, value in sorted(vars(args).items()):
        if key.startswith("_"):
            continue
        print(f"  {key}: {value}", flush=True)


def _cache_name(prefix, metadata):
    payload = json.dumps(metadata, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.pt"


def feature_cache_path(args, num_hops, max_hops, extra_metapath):
    if not args.cache_propagation:
        return None
    gsmp_mode = "baseline"
    if getattr(args, "smp_first_layer", False):
        gsmp_mode = (
            f"smp-{args.gsmp_scope}-{args.gsmp_time_source}-"
            f"{args.gsmp_derived_time}"
        )
    elif args.gsmp_first_layer:
        gsmp_mode = (
            f"gsmp-{args.gsmp_scope}-{args.gsmp_normalizer}-"
            f"{args.gsmp_time_source}-{args.gsmp_derived_time}"
        )
    metadata = {
        "dataset": args.dataset,
        "kind": "feature",
        "root": str(Path(args.root).resolve()),
        "extra_embedding": args.extra_embedding,
        "emb_path": str(Path(args.emb_path).resolve()),
        "num_hops": num_hops,
        "max_hops": max_hops,
        "extra_metapath": extra_metapath,
        "gsmp_mode": gsmp_mode,
    }
    cache_dir = Path(args.propagation_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _cache_name("feature_prop", metadata)


def save_cache_atomic(obj, path):
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def result_line(method, seed, stage, epoch, train_acc, val_acc, test_acc,
                best_epoch, best_val, best_test, elapsed_sec):
    return (
        f"RESULT method={method} seed={seed} stage={stage} epoch={epoch} "
        f"train_acc={train_acc:.8f} val_acc={val_acc:.8f} test_acc={test_acc:.8f} "
        f"best_epoch={best_epoch} best_val={best_val:.8f} "
        f"best_test_at_best_val={best_test:.8f} elapsed_sec={elapsed_sec:.2f}"
    )


def append_progress(path, line):
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if not progress_path.exists():
        progress_path.write_text(
            "timestamp\tmethod\tseed\tstage\tepoch\ttrain_acc\tval_acc\ttest_acc\t"
            "best_epoch\tbest_val\tbest_test_at_best_val\telapsed_sec\n"
        )
    fields = {}
    for item in line.split()[1:]:
        key, value = item.split("=", 1)
        fields[key] = value
    with open(progress_path, "a") as f:
        f.write(
            f"{datetime.datetime.now().isoformat(timespec='seconds')}\t"
            f"{fields['method']}\t{fields['seed']}\t{fields['stage']}\t{fields['epoch']}\t"
            f"{fields['train_acc']}\t{fields['val_acc']}\t{fields['test_acc']}\t"
            f"{fields['best_epoch']}\t{fields['best_val']}\t"
            f"{fields['best_test_at_best_val']}\t{fields['elapsed_sec']}\n"
        )


def main(args):
    if args.seed >= 0:
        set_random_seed(args.seed)
    print_resolved_config(args)
    run_tic = time.time()
        
    g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset(args)
    gsmp_time_dict = getattr(g, '_gsmp_time_dict', None)

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

        extra_metapath = [] # ['AIAP', 'PAIAP']
        extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

        print(f'Current num hops = {args.num_hops}')
        if len(extra_metapath):
            max_hops = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_hops = args.num_hops + 1

        cache_path = feature_cache_path(args, args.num_hops, max_hops, extra_metapath)
        if cache_path is not None and cache_path.exists():
            print(f'Load cached feature propagation from {cache_path}', flush=True)
            feats = torch.load(cache_path, map_location='cpu')
            g = clear_hg(g, echo=False)
        else:
            # compute k-hop feature
            g = hg_propagate(
                g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False,
                gsmp_first_layer=(args.gsmp_first_layer or args.smp_first_layer),
                gsmp_time_dict=gsmp_time_dict,
                gsmp_normalizer=args.gsmp_normalizer,
                gsmp_scope=args.gsmp_scope,
                is_label_propagation=False,
                args=args,
                temporal_first_layer_method=args.temporal_first_layer_method)

            feats = {}
            keys = [k for k in list(g.nodes[tgt_type].data.keys()) if is_metapath_key(k)]
            print(f'Involved feat keys {keys}')
            for k in keys:
                feats[k] = g.nodes[tgt_type].data.pop(k)

            g = clear_hg(g, echo=False)
            if cache_path is not None:
                save_cache_atomic(feats, cache_path)
                print(f'Saved feature propagation cache to {cache_path}', flush=True)
    else:
        assert 0

    feats = {k: v[init2sort] for k, v in feats.items()}

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
                    gsmp_first_layer=(args.gsmp_first_layer or args.smp_first_layer),
                    gsmp_time_dict=gsmp_time_dict,
                    gsmp_normalizer=args.gsmp_normalizer,
                    gsmp_scope=args.gsmp_scope,
                    is_label_propagation=True,
                    gsmp_apply_label_prop=args.gsmp_apply_label_prop,
                    args=args,
                    temporal_first_layer_method=args.temporal_first_layer_method)

                keys = [k for k in list(g.nodes[tgt_type].data.keys()) if is_metapath_key(k)]
                print(f'Involved label keys {keys}')
                for k in keys:
                    if k == tgt_type: continue
                    label_feats[k] = g.nodes[tgt_type].data.pop(k)
                g = clear_hg(g, echo=False)

                # label_feats = remove_self_effect_on_label_feats(label_feats, label_onehot)
                for k in ['PPP', 'PAP', 'PFP', 'PPPP', 'PAPP', 'PPAP', 'PFPP', 'PPFP']:
                    if k in label_feats:
                        diag = torch.load(os.path.join(args.emb_path, f'{args.dataset}_{k}_diag.pt'))
                        label_feats[k] = label_feats[k] - diag.unsqueeze(-1) * label_onehot
                        assert torch.all(label_feats[k] > -1e-6)
                        # print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())

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
            should_stop = False
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

                if val_acc > best_val_acc:
                    best_epoch = epoch
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                    torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                    count = 0
                else:
                    count = count + args.eval_every
                    if count >= args.patience:
                        should_stop = True
                line = result_line(
                    args.method_name, args.seed, stage, epoch, acc,
                    val_acc, test_acc, best_epoch, best_val_acc,
                    best_test_acc, time.time() - run_tic)
                append_progress(args.progress_file, line)
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}\n{}".format(
                    best_epoch, best_val_acc*100, best_test_acc*100, line)
            print(log, flush=True)
            if should_stop:
                break

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
    ## GSMP feature propagation
    parser.add_argument("--gsmp-first-layer", action='store_true', default=False,
                        help="enable GSMP feature propagation; scope is controlled by --gsmp-scope")
    parser.add_argument("--smp-first-layer", action='store_true', default=False,
                        help="enable SMP feature propagation at the same guarded insertion point as --gsmp-first-layer")
    parser.add_argument("--gsmp-normalizer", choices=("nonempty", "global"), default="nonempty",
                        help="nonempty preserves mean scale over observed source-time bins")
    parser.add_argument("--gsmp-scope", choices=("paper-stack", "first-hop"), default="paper-stack",
                        help="paper-stack applies GSMP to every eligible direct P-P feature propagation step")
    parser.add_argument("--gsmp-time-source", choices=("all", "train-only"), default="all",
                        help="which paper years are allowed for deriving node time bins")
    parser.add_argument("--gsmp-derived-time", choices=("mode",), default="mode",
                        help="derived timestamp rule for non-paper nodes")
    parser.add_argument("--gsmp-cache-dir", type=str, default="./cache/gsmp",
                        help="cache directory for per-edge GSMP weights")
    parser.add_argument("--gsmp-apply-label-prop", action='store_true', default=False,
                        help="optional ablation: also apply GSMP to label propagation using the selected GSMP scope")
    parser.add_argument("--no-gsmp-cache", action='store_true', default=False,
                        help="debugging only: recompute GSMP edge weights")
    ## Runtime accounting and cache controls
    parser.add_argument("--propagation-cache-dir", type=str, default="./cache/propagation",
                        help="cache directory for seed-independent propagated feature tensors")
    parser.add_argument("--cache-propagation", action='store_true', default=True,
                        help="cache seed-independent feature propagation tensors")
    parser.add_argument("--no-cache-propagation", dest="cache_propagation", action='store_false',
                        help="disable feature propagation cache")
    parser.add_argument("--progress-file", type=str, default="./results/live_progress.tsv",
                        help="append-only TSV with validation/test progress")
    parser.add_argument("--method-name", type=str, default=None,
                        help="result label; defaults to baseline or gsmp")
    parser.add_argument("--gc-every", type=int, default=1,
                        help="run Python garbage collection every N epochs; 0 disables it")
    parser.add_argument("--empty-cache-every-epoch", action='store_true', default=False,
                        help="call torch.cuda.empty_cache around every epoch; slower but useful for tight memory")

    return parser.parse_args(args)


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    assert args.dataset.startswith('ogbn')
    if args.gsmp_first_layer and args.smp_first_layer:
        raise ValueError("--gsmp-first-layer and --smp-first-layer are mutually exclusive")
    if args.gsmp_apply_label_prop and not (args.gsmp_first_layer or args.smp_first_layer):
        raise ValueError("--gsmp-apply-label-prop requires --gsmp-first-layer or --smp-first-layer")
    args.temporal_first_layer_method = "smp" if args.smp_first_layer else "gsmp"
    if args.method_name is None:
        if args.smp_first_layer:
            args.method_name = "smp"
        else:
            args.method_name = "gsmp" if args.gsmp_first_layer else "baseline"
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
