"""Test T8: losses, ordinal heads (coral, coral_src), multi-source sampler, SAM optimizer."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.samplers import JointBatchSampler, MultiSourceBatchSampler
from src.losses.losses import coral_loss, emd2, total_loss
from src.models.gradenet import GradeNet, SourceOrdinalHead, probs_from_logits
from src.training.sam import SAM


def test_emd2_zero_and_ordinal_monotonicity():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    z_perfect = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0]])
    assert pytest.approx(emd2(z_perfect, q).item(), abs=1e-5) == 0.0
    losses = []
    for c in range(1, 5):
        z = torch.full((1, 5), -100.0)
        z[0, c] = 100.0
        losses.append(emd2(z, q).item())
    assert all(a < b for a, b in zip(losses, losses[1:]))


@pytest.mark.parametrize('head', ['coral', 'coral_src'])
def test_ordinal_probabilities_sum_to_one(head):
    z = torch.randn(7, 4)
    p = probs_from_logits(z, head)
    assert torch.allclose(p.sum(1), torch.ones(7), atol=1e-5)
    assert (p >= 0).all()


def test_coral_loss_weights_reduce_to_mean():
    z, y = torch.randn(6, 4), torch.randint(0, 5, (6,))
    assert torch.allclose(coral_loss(z, y), coral_loss(z, y, w=torch.ones(6)), atol=1e-6)
    w = torch.tensor([1.0, 0, 0, 0, 0, 0])
    single = coral_loss(z[:1], y[:1])
    assert torch.allclose(coral_loss(z, y, w=w), single, atol=1e-6)


def test_source_head_ordering_offsets_and_routing():
    torch.manual_seed(0)
    head = SourceOrdinalHead(in_dim=8, K=5, n_sources=3, dropout=0.0, use_offsets=True)
    with torch.no_grad():
        head.delta.copy_(torch.randn(3, 4) * 2.0)
        head.raw.copy_(torch.randn(4))
    for idx in (None, torch.tensor([0, 1, 2, -1])):
        b = head.thresholds(idx)
        assert (b[:, 1:] > b[:, :-1]).all(), 'cut-points must stay ordered for any parameters'
    shared = head.thresholds()
    unseen = head.thresholds(torch.tensor([-1]))
    assert torch.allclose(shared, unseen), 'non-source images use the shared cut-points'
    f = torch.randn(4, 8)
    z = head(f, torch.tensor([0, 1, 2, -1]))
    s = head.score(f)
    assert torch.allclose(z, s - head.thresholds(torch.tensor([0, 1, 2, -1])))
    assert head.offset_penalty().item() > 0
    head2 = SourceOrdinalHead(8, 5, 3, 0.0, use_offsets=False)
    assert head2.delta is None and head2.offset_penalty().item() == 0.0


def test_gradenet_coral_src_domains_and_thresholds():
    cfg = {'backbone': 'resnet18', 'pretrained': False, 'head_type': 'coral_src', 'num_classes': 5,
           'extra_sources': ['ddr', 'deepdrid'], 'src_offsets': True, 'head_dropout': 0.0, 'drop_path': 0.0}
    m = GradeNet(cfg).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        m.head.delta.add_(1.0)
        z_shared = m(x)
        z_src = m(x, domain=torch.tensor([0, 3, 2]))   # aptos, ddr, messidor (not a source)
    assert z_shared.shape == (3, 4)
    assert not torch.allclose(z_src[0], z_shared[0]) and not torch.allclose(z_src[1], z_shared[1])
    assert torch.allclose(z_src[2], z_shared[2]), 'an evaluation-only dataset must use the shared cut-points'
    thr = m.source_thresholds()
    assert set(thr) == {'shared', 'aptos', 'ddr', 'deepdrid'}
    # old checkpoints (no src_lut in the state dict) still load into softmax / coral GradeNets
    for head in ('softmax', 'coral'):
        g = GradeNet({**cfg, 'head_type': head})
        assert 'src_lut' not in g.state_dict()


def test_total_loss_coral_src_trains():
    torch.manual_seed(0)
    cfg = {'head_type': 'coral_src', 'num_classes': 5}
    head = SourceOrdinalHead(4, 5, 2, 0.0, True)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    f = torch.randn(64, 4)
    y = torch.clamp((f[:, 0] * 1.5 + 2).round().long(), 0, 4)
    src = torch.randint(0, 2, (64,))
    first = None
    for _ in range(100):
        opt.zero_grad()
        loss = total_loss(head(f, src), None, y, cfg) + 0.1 * head.offset_penalty()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first


def test_multisource_sampler_batch_layout():
    s = MultiSourceBatchSampler(n_a=20, source_labels=[np.arange(30) % 5, np.arange(9) % 5], B=8, seed=0)
    batches = list(s)
    assert len(batches) == len(s) == 20 // 4
    seen_aptos = []
    for b in batches:
        assert len(b) == 8
        aptos = [i for i in b if i < 20]
        src1 = [i for i in b if 20 <= i < 50]
        src2 = [i for i in b if i >= 50]
        assert len(aptos) == 4 and len(src1) == 2 and len(src2) == 2
        assert max(b) < 59
        seen_aptos += aptos
    assert len(set(seen_aptos)) == len(seen_aptos), 'APTOS is drawn without replacement'
    j = list(JointBatchSampler(20, np.arange(30) % 5, 8, seed=0))
    assert all(len(b) == 8 for b in j)


def test_sam_optimizer():
    torch.manual_seed(42)
    model = nn.Linear(10, 2)
    optimizer = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    x, y = torch.randn(16, 10), torch.randint(0, 2, (16,))
    crit = nn.CrossEntropyLoss()
    initial = crit(model(x), y).item()

    def closure():
        optimizer.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    assert crit(model(x), y).item() < initial


def test_llrd_groups_cover_every_parameter_once():
    from src.models.gradenet import llrd_groups
    for cfg, decay in (({'backbone': 'vit_tiny_patch16_224', 'backbone_global_pool': 'avg'}, 0.65),
                       ({'backbone': 'resnet18'}, 0.8),
                       ({'backbone': 'resnet18', 'head_type': 'coral_src', 'extra_sources': ['ddr']}, 0.8)):
        m = GradeNet({'pretrained': False, 'head_type': 'softmax', 'drop_path': 0.0, 'head_dropout': 0.0, **cfg})
        groups = llrd_groups(m, 1e-3, decay, 0.05, 10)
        ids = [id(p) for g in groups for p in g['params']]
        assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    vit = GradeNet({'pretrained': False, 'backbone': 'vit_tiny_patch16_224', 'head_type': 'softmax', 'drop_path': 0.0})
    g = llrd_groups(vit, 1e-3, 0.65, 0.05, 10)
    name_of = {id(p): n for n, p in vit.backbone.named_parameters()}
    lr_of = {name_of[id(p)]: gr['lr'] for gr in g for p in gr['params'] if id(p) in name_of}
    assert lr_of['patch_embed.proj.weight'] < lr_of['blocks.0.attn.qkv.weight'] < lr_of['blocks.11.attn.qkv.weight']


def test_pretrain_loader_accepts_mae_checkpoints(tmp_path):
    import timm
    b = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    torch.save({'model': {**b.state_dict(), 'head.weight': torch.zeros(5, 192)}}, tmp_path / 'rf.pth')
    m = GradeNet({'pretrained': False, 'backbone': 'vit_tiny_patch16_224', 'backbone_global_pool': 'avg',
                  'head_type': 'softmax', 'pretrain_path': str(tmp_path / 'rf.pth')})
    assert torch.allclose(m.backbone.blocks[0].attn.qkv.weight, b.blocks[0].attn.qkv.weight)
    small = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0, img_size=112)
    torch.save({'model': small.state_dict()}, tmp_path / 'bad.pth')
    with pytest.raises(ValueError, match='shape mismatch'):
        GradeNet({'pretrained': False, 'backbone': 'vit_tiny_patch16_224', 'head_type': 'softmax',
                  'pretrain_path': str(tmp_path / 'bad.pth')})
