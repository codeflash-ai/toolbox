import warnings

def neutralize(weights, assets, group = 'market'):
    """
    :param weights: xarray with weights of the algorithm
    :param assets: qndata.load_assets
    :param group: neutralize positions by 'market', 'industry' or 'sector'
    :return: xarray with neutrlized positions
    """
    result = weights.copy(True)

    if group in ['industry','sector']:
        # Build asset id set for fast lookup
        weights_asset_set = set(weights.asset.values)
        filtered_assets = []
        asset_names_set = set()
        for a in assets:
            if a['id'] in weights_asset_set:
                filtered_assets.append(a)
                asset_names_set.add(a['id'])
        no_info_assets = [a for a in weights.asset.values if a not in asset_names_set]

        # Collect groupings
        from collections import defaultdict
        groups = defaultdict(list)
        for a in filtered_assets:
            g = a.get(group)
            groups[g].append(a['id'])
        
        if no_info_assets:
            groups['no_info'] = no_info_assets
            warnings.warn("Some stocks has no specification. Perhaps you are using illiquid instruments or outdated assets data.")
        
        # Local variable to avoid repeated attribute lookups
        _sel = result.sel
        _result_loc = result.loc
        # Precompute mean for all relevant groups and slice assign in batch
        for j, asset_list in groups.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                group_sel = _sel(asset=asset_list)
                mean_val = group_sel.mean('asset')
                _result_loc[{'asset': asset_list}] = group_sel - mean_val

    elif group == 'market':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = result - result.mean('asset')
    else:
        raise Exception(f"No such group '{group}'. Use 'market', 'sector' or 'industry' instead.")

    return result
