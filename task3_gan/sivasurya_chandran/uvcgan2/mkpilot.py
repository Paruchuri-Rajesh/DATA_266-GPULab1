import yaml, copy
c = yaml.safe_load(open("uvcgan2/config_uvcgan.yaml"))
p = copy.deepcopy(c)
p["train"]["out_dir"] = "uvcgan2/pilot_checkpoints"
p["train"]["log_file"] = "uvcgan2/logs/train_raw_pilot.log"
p["train"]["min_epochs"] = 6
p["train"]["sample_every"] = 500          # frequent samples so we can eyeball structure
p["train"]["init_generators_from"] = "uvcgan2/checkpoints/pretrain/{name}.pt"
p["select"]["n_evals"] = 4
p["select"]["from_frac"] = 0.15
p["eval"]["pred_dir"] = "uvcgan2/pilot_outputs"
yaml.safe_dump(p, open("uvcgan2/config_pilot.yaml", "w"), sort_keys=False)
print("wrote uvcgan2/config_pilot.yaml")
print("  out_dir      :", p["train"]["out_dir"])
print("  min_epochs   :", p["train"]["min_epochs"])
print("  n_evals      :", p["select"]["n_evals"])
print("  sample_every :", p["train"]["sample_every"])
