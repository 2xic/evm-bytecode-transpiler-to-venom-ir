Found out the issue with the phi functions
- https://github.com/vyperlang/vyper/pull/4475

```
function global {
  global:
    jnz 1, @1_then, @2_then
  1_then:
    %1 = 1
    jmp @3_phi
  2_then:
    %2 = 2
    jmp @3_phi
  3_phi:
    %3 = phi @1_then, %1, @2_then, %2
    return 0,%3
}
```

So we need to be able to use the phi functions instead of doing that silly duplicating of multiple blocks that we did before. So we need to keep track of the parent blocks and assign the ids. The phi function can then be constructed from this info.
