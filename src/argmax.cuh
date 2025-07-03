#pragma once


struct MaskedValue {
    int index;
    float value;
    bool valid;

    MaskedValue();
    MaskedValue(int i, float v, bool m);
};

struct ArgmaxContext {
    MaskedValue *zipped = nullptr;
    MaskedValue *output = nullptr;
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    void init(const size_t size);
    void free();
    ~ArgmaxContext() {
        free();
    }
};
