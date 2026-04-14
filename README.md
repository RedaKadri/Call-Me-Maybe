# stupid code

```py
def stuff(self, func_call, logits):
    allowed_ids = self.model.encode('{"')

    constrained_logits = np.full_like(logits, -np.inf)
    constrained_logits[allowed_ids] = logits[allowed_ids]
```
