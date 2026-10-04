import yaml, copy
c = yaml.safe_load(open("uvcgan2/config_pilot.yaml"))
p = copy.deepcopy(c)
p["train"]["lambda_lowres"] = 0.5
p["train"]["lambda_identity"] = 0.5
p["train"]["lambda_cycle"] = 10.0
p["train"]["out_dir"] = "uvcgan2/pilot2_checkpoints"
p["train"]["log_file"] = "uvcgan2/logs/train_raw_pilot2.log"
p["eval"]["pred_dir"] = "uvcgan2/pilot2_outputs"
yaml.safe_dump(p, open("uvcgan2/config_pilot2.yaml", "w"), sort_keys=False)
print("wrote uvcgan2/config_pilot2.yaml")
for k in ("lambda_cycle","lambda_identity","lambda_lowres","out_dir","init_generators_from"):
    print("  %-22s %s" % (k, p["train"][k]))
