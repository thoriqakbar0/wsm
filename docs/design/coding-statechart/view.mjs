export const state = v => v.phase;
export const title = 'Python edit execution and checkpoint publication';
export const step = action => ({Run:'Execute Python', Finish:'Program succeeds', Verify:'Reload and verify checkpoint', Publish:'Publish checkpoint atomically', Fail:'Execution or verification fails', Interrupt:'Interrupt before publication', Done:null}[action]);
