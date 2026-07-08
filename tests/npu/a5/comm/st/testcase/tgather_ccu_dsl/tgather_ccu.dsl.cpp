#include "stdcc.h"

// First-version 2-rank gather data path:
// - non-root rank sends one 16B payload to root
// - root receives the peer payload into the output slice passed by host
void tgather_ccu_dsl_tx(void *data)
{
    SMA s(16);
    void *payload = load(data, 16);
    write(s, 0, payload);
}

void tgather_ccu_dsl_rx(void *data)
{
    SMA s(16);
    s.wait();
    store(data, read(s, 0), 16);
}
