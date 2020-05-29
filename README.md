# pytorch-neural-style-transfer
Command line application to run a neural-style transfer using PyTorch

```python
python main.py --help

Usage: main.py [OPTIONS]

Options:
  -c, --content TEXT      path to content image
  -s, --style TEXT        path to style image
  -o, --output TEXT       path to save generated image
  --style_weight INTEGER  [default: 1000000]
  --steps INTEGER         [default: 300]
  --random_input          Will start with random noise if set, otherwise
                          content image

  --help                  Show this message and exit.
```

### Example:
  
`python main.py --content='./examples/bootsy.jpg' --style='./examples/picasso.jpg' --style_weight=500000 --steps=3000 --random_input`

![Image of Picasso Bootsy](https://github.com/tomsitter/pytorch-neural-style-transfer/blob/master/output/bootsy_picasso_3000_steps.png)
  
  
### Other Example:
![Image of Muted Picasso Booty](https://github.com/tomsitter/pytorch-neural-style-transfer/blob/master/output/bootsy_final.jpg)
  
