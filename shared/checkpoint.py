#!/usr/bin/env python
# coding=utf-8
"""Safe checkpoint loading utility."""
import logging
import torch

DEFAULT_CRITICAL_PATTERNS = ['fc.', 'vaemodel.', 'model.', 'base_conv.']

def load_state_dict_safe(model, state_dict, strict=False,
                         critical_patterns=None, model_name='model'):
    if critical_patterns is None:
        critical_patterns = DEFAULT_CRITICAL_PATTERNS
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logging.warning('[%s] Missing keys: %d total', model_name, len(missing))
        for pat in critical_patterns:
            cm = [k for k in missing if pat in k]
            if cm:
                logging.error('[%s] CRITICAL: pattern "%s" MISSING (%d): %s',
                              model_name, pat, len(cm), cm[:5])
    else:
        logging.info('[%s] All keys loaded successfully.', model_name)
    if unexpected:
        logging.info('[%s] Unexpected keys (ignored): %d', model_name, len(unexpected))
    return missing, unexpected

def convert_fc_1to2(state_dict, fc_weight_key='fc.5.weight', fc_bias_key='fc.5.bias'):
    """DEPRECATED (P0-7): legacy 1->2 output conversion. Not equivalent to
    real EDL training; all call sites now refuse legacy 1-output checkpoints
    instead. Retained only for inspecting legacy checkpoints."""
    if fc_weight_key in state_dict and state_dict[fc_weight_key].shape[0] == 1:
        logging.warning('WARNING: 1->2 output conversion (NOT equivalent to real EDL training)')
        ow, ob = state_dict[fc_weight_key], state_dict[fc_bias_key]
        state_dict[fc_weight_key] = torch.cat([ow, -ow], dim=0)
        state_dict[fc_bias_key] = torch.cat([ob, -ob], dim=0)
        return True
    return False

def log_seed_checkpoint_note(checkpoint_path, seeds):
    logging.warning('NOTE: Same checkpoint (%s) across %d seeds. SD = neg-sampling variation.',
                    checkpoint_path, len(seeds))
