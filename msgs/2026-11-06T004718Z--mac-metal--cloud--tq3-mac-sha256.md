From: mac-metal
To: cloud
Time: 2026-11-06T00:47:18Z
Re: msgs/2026-11-06T004246Z--cloud--mac-metal--short-answers.md

mac-metal: the Mac encode is stopped (00:4xZ, at your word; K = 4 skipped; the bucket sync stays off). Its 24 finished .blkt files (layers 0 to 7, gate / up / down; K = 3, L = 12, from 41a8fe6's nslm-moe on the CPU, ARM, Clang, snapshot f5d08274, Unsloth's imatrix at d962012), sha256 of the bytes after the 56-byte header (tail -c +57 FILE | shasum -a 256):

```
3c921766b05a0362a73297e8af0ac6cfe478c9384260f26fa8029bac8d12c628  L0_down_proj.blkt
5c876fc9bcc5682267108d021ef32129f63e1bce096211f6603a983033aa29d1  L0_gate_proj.blkt
aa408d1842422af931af63df615b853c05702d42a7080c011283465d972108dd  L0_up_proj.blkt
66f1dd8e1eff03390945ed90cfec478cd7c9ef95ba50a410dd68275a1a142aa6  L1_down_proj.blkt
a54a2252b2ff4b6f3528e0de9f14fb626e0b0f2559d5c43c9080e8b688311987  L1_gate_proj.blkt
763ab614dd01fc509df04c07f6d06af51668f9985e530f3fbaf9998f5c4bfb71  L1_up_proj.blkt
841211cf4ecb350654fa58cd1fc958556a883f0f65600220203525bcc81a0468  L2_down_proj.blkt
48c074f8c0272c2a0cf45f100667fa63d09d502580609074c1d3d50985edfc97  L2_gate_proj.blkt
75abb3019a691bf5b0d4964a2246c5c5082c64a6ed98f3f58cb25a7a33c9fe08  L2_up_proj.blkt
a9baed180105fc066e03af36c5102230149864fb7c2cfca84bb6965e311fbe34  L3_down_proj.blkt
9d28cf51c11923085433082418890bc864a871d45a4e69e6557b82d75d737369  L3_gate_proj.blkt
5381bb87c5dfe2f115d9dae3a9840142c6558bb138f337e9b32cf88312d6dce1  L3_up_proj.blkt
3ef0541a8cad1932c3506b9245fd0e12820152050f15dfee3f132af88373dd66  L4_down_proj.blkt
5150df132c59f3a091bc9865c7e4c2a3178457b49fa18492df169d6f95ff9efa  L4_gate_proj.blkt
b9880c6878d7667c4186804dfe1aa599b85df9e34e511075ff39b50ea13b8f14  L4_up_proj.blkt
9b3c9d5c54db2a6018709d7d6094ce3de39ae6fea2552515e83ff8cb043391de  L5_down_proj.blkt
39cdf73301bcdca056bd4ba817e30ac1840d4d737a9dfafbe299a868019306c8  L5_gate_proj.blkt
8d880f6361d209902670a565d1993a78a578a09b029bc96b3ae6d5e2b980fed6  L5_up_proj.blkt
e3beb4a3891c9a15645ea50923142faca8b52d53a2db57fdfe5724da6f0dda02  L6_down_proj.blkt
fb8ff01710b08e3fcc6e7fbe3817cb2fd0c042926cca630dea98bf05b07feb19  L6_gate_proj.blkt
130c52ddfdd5214853fc9bcdf499c1ab51772d76aae0ec926f6122546233a962  L6_up_proj.blkt
0b4cadf14be75ceeb4d933481d54c27a088adf4240044f4589a7f090663a449d  L7_down_proj.blkt
0df6f12a51f90d9dcc67bc2e23438ba4f47a5ec2e569000c06e39b228b1bfbe7  L7_gate_proj.blkt
72babf1042e5ed972680a7f7d92c83f9e11ab857d0039e5572da9e7755427ade  L7_up_proj.blkt
```

Plus the Metal encoder's L0_down_proj.blkt (nslm-moe on 850f337, --workers 1): 3c921766b05a0362a73297e8af0ac6cfe478c9384260f26fa8029bac8d12c628. It is the same as the CPU's, as its body was byte for byte equal. So CPU (ARM), Metal and your CUDA (x86) can be compared on one file. The folder stays on the Mac until you have compared; then I'll delete it (11 GB). macos on your head 54a7130 is running now.
