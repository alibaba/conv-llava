# Model Zoo

## Performance

<table class="tg"><thead>
  <tr>
    <th class="tg-nrix">Method</th>
    <th class="tg-nrix">Resolution</th>
    <th class="tg-nrix">Visual Tokens</th>
    <th class="tg-nrix">LLM</th>
    <th class="tg-nrix">MME</th>
    <th class="tg-nrix">MMB</th>
    <th class="tg-nrix">SEED</th>
    <th class="tg-nrix">RealWorldQA</th>
    <th class="tg-nrix">MMMU</th>
    <th class="tg-nrix">MMVet</th>
    <th class="tg-nrix">Text</th>
    <th class="tg-nrix">Doc</th>
    <th class="tg-nrix">POPE</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">768</td>
    <td class="tg-nrix">144</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">1541</td>
    <td class="tg-nrix">68</td>
    <td class="tg-nrix">68.8</td>
    <td class="tg-nrix">55.9</td>
    <td class="tg-nrix">36.3</td>
    <td class="tg-nrix">44.8</td>
    <td class="tg-nrix">59.1</td>
    <td class="tg-nrix">44.8</td>
    <td class="tg-nrix">87.3</td>
  </tr>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">1024</td>
    <td class="tg-nrix">256</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">1553</td>
    <td class="tg-nrix">68.8</td>
    <td class="tg-nrix">69.3</td>
    <td class="tg-nrix">58.8</td>
    <td class="tg-nrix">35.1</td>
    <td class="tg-nrix">44.4</td>
    <td class="tg-nrix">62.5</td>
    <td class="tg-nrix">48.5</td>
    <td class="tg-nrix">87.7</td>
  </tr>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">1536</td>
    <td class="tg-nrix">576</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">1575</td>
    <td class="tg-nrix">68.7</td>
    <td class="tg-nrix">70.2</td>
    <td class="tg-nrix">59.9</td>
    <td class="tg-nrix">35.8</td>
    <td class="tg-nrix">45.9</td>
    <td class="tg-nrix">65.8</td>
    <td class="tg-nrix">59</td>
    <td class="tg-nrix">87.3</td>
  </tr>
</tbody></table>

<table class="tg"><thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Method</th>
    <th class="tg-nrix" rowspan="2">Resolution</th>
    <th class="tg-nrix" rowspan="2">Visual Tokens</th>
    <th class="tg-nrix" rowspan="2">LLM</th>
    <th class="tg-nrix" colspan="3">RefCOCO</th>
    <th class="tg-nrix" colspan="3">RefCOCO+</th>
    <th class="tg-nrix" colspan="2">RefCOCOg</th>
    <th class="tg-nrix" rowspan="2">Avg</th>
  </tr>
  <tr>
    <th class="tg-nrix">val</th>
    <th class="tg-nrix">test-A</th>
    <th class="tg-nrix">test-B</th>
    <th class="tg-nrix">val</th>
    <th class="tg-nrix">test-A</th>
    <th class="tg-nrix">test-B</th>
    <th class="tg-nrix">val</th>
    <th class="tg-nrix">test</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">768</td>
    <td class="tg-nrix">144</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">84.5</td>
    <td class="tg-nrix">89.0</td>
    <td class="tg-nrix">79.2</td>
    <td class="tg-nrix">77.7</td>
    <td class="tg-nrix">84.9</td>
    <td class="tg-nrix">69.7</td>
    <td class="tg-nrix">79.8</td>
    <td class="tg-nrix">79.7</td>
    <td class="tg-nrix">80.6</td>
  </tr>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">1024</td>
    <td class="tg-nrix">256</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">85.5</td>
    <td class="tg-nrix">89.6</td>
    <td class="tg-nrix">78.8</td>
    <td class="tg-nrix">79.3</td>
    <td class="tg-nrix">86.1</td>
    <td class="tg-nrix">70.3</td>
    <td class="tg-nrix">80.6</td>
    <td class="tg-nrix">81.2</td>
    <td class="tg-nrix">81.4</td>
  </tr>
  <tr>
    <td class="tg-nrix">ConvLLaVA</td>
    <td class="tg-nrix">1536</td>
    <td class="tg-nrix">576</td>
    <td class="tg-nrix">7B</td>
    <td class="tg-nrix">86.5</td>
    <td class="tg-nrix">90.6</td>
    <td class="tg-nrix">80.5</td>
    <td class="tg-nrix">80.0</td>
    <td class="tg-nrix">86.8</td>
    <td class="tg-nrix">71.5</td>
    <td class="tg-nrix">82.0</td>
    <td class="tg-nrix">82.4</td>
    <td class="tg-nrix">82.3</td>
  </tr>
</tbody></table>

## Download

We release checkpoints after vision language pretraining and visual instruction tuning. You could directly use the sft model and finetune the vision language pretraining checkpoints on you own data.

|     model      |                                  Huggingface                                  |                                                                          ModelScope                                                                          | WiseModel                                                                             |
| :------------: | :---------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------- |
| ConvLLaVA-768  | [pretrain](https://huggingface.co/ConvLLaVA/ConvLLaVA-pretrain-768), [sft](https://huggingface.co/ConvLLaVA/ConvLLaVA-sft-768)  |  [pretrain](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-pretrain-768/summary), [sft](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-sft-768/summary)  | [pretrain](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-pretrain-768/intro), [sft](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-sft-768/intro) |
| ConvLLaVA-1024 | [pretrain](https://huggingface.co/ConvLLaVA/ConvLLaVA-pretrain-1024), [sft](https://huggingface.co/ConvLLaVA/ConvLLaVA-sft-1024) | [pretrain](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-pretrain-1024/summary), [sft](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-sft-1024/summary) | [pretrain](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-pretrain-1024/intro), [sft](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-sft-1024/intro) |
| ConvLLaVA-1536 | [pretrain](https://huggingface.co/ConvLLaVA/ConvLLaVA-pretrain-1536), [sft](https://huggingface.co/ConvLLaVA/ConvLLaVA-sft-1536) | [pretrain](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-pretrain-1536/summary), [sft](https://modelscope.cn/models/ConvLLaVA/ConvLLaVA-sft-1536/summary) | [pretrain](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-pretrain-1536/intro), [sft](https://wisemodel.cn/models/ConvLLaVA/ConvLLaVA-sft-1536/intro) |

The **pretrain** above means the checkpoints are after the second stage **vision-language pretraining**. The **sft** above means the checkpoints are after the third stage **instruction tuning**.

## Usage of the scripts

The three stages training scripts are listed below:

- Projector Initialzation: [stage1](https://github.com/alibaba/conv-llava/tree/main/scripts/stage_1.sh)
- Vision Language Pretraining: [stage2](https://github.com/alibaba/conv-llava/tree/main/scripts/stage_2.sh)
- Instruction Tuning: [stage3](https://github.com/alibaba/conv-llava/tree/main/scripts/stage_3.sh)

If you want to custimze your model, you can directly load the **second stage pretrained visual encoder and LLM** for instruction tuning. It takes about 6 hours to train the 768 resolution model with LLaVA-Instruct-665k on 8 A800 GPUs.

If you wang to train from scratch, you could download our processed ConvNeXt model (modify from LAION ConvNeXt). Then follow the three stage training scripts to train the model.
