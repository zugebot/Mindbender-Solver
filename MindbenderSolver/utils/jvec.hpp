#pragma once

void _JVEC_HIDDEN_PRINTF(const char* str, unsigned long long num);
void _JVEC_HIDDEN_MEMCPY(void *dst, const void *src, unsigned long long size);


template<class T>
class [[maybe_unused]] JVec {
    void* myRawMemory{};
    unsigned long long mySize{};
    unsigned long long myCapacity{};

    inline T* ptr_t() { return static_cast<T*>(myRawMemory); }

public:
    [[maybe_unused]] explicit JVec(unsigned long long theCapacity);
    ~JVec();

    inline unsigned long long size() { return mySize; }
    inline unsigned long long capacity() { return myCapacity; }

    inline T* data() { return ptr_t(); }
    inline T* begin() { return ptr_t(); }
    inline T* end() { return ptr_t() + mySize; }

    inline void clear() { mySize = 0; }
    void resize(unsigned long long theSize);
    void reserve(unsigned long long theCapacity);
    void swap(JVec& other);

    inline T& operator[](unsigned long long index) { return ptr_t()[index]; }
    inline const T& operator[](unsigned long long index) const { return ptr_t()[index]; }

    JVec(const JVec&) = delete;
    JVec& operator=(const JVec&) = delete;
    JVec(JVec&&) = delete;
    JVec& operator=(JVec&&) = delete;
};


template<class T>
[[maybe_unused]] JVec<T>::JVec(unsigned long long theCapacity) {
    mySize = 0;
    myCapacity = theCapacity;
    myRawMemory = operator new[](theCapacity * sizeof(T));
    if (!myRawMemory) {
        _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %d", theCapacity);
    }
}


template<class T>
[[maybe_unused]] JVec<T>::~JVec() {
    operator delete[](myRawMemory);
    myRawMemory = nullptr;
    mySize = 0;
    myCapacity = 0;
}


template<class T>
void JVec<T>::resize(unsigned long long theSize) {
    if (theSize < myCapacity) {
        mySize = theSize;
    } else {
        void* newPtr = operator new[](theSize * sizeof(T));
        if (!myRawMemory) {
            _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %d", theSize);
        }
        _JVEC_HIDDEN_MEMCPY(newPtr, myRawMemory, mySize * sizeof(T));
        operator delete[](myRawMemory);
        myRawMemory = newPtr;
        mySize = theSize;
        myCapacity = theSize;
    }
}


template<class T>
void JVec<T>::reserve(unsigned long long theCapacity) {
    if (theCapacity > myCapacity) {
        void* newPtr = operator new[](theCapacity * sizeof(T));
        if (!myRawMemory) {
            _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %d", theCapacity);
        }
        _JVEC_HIDDEN_MEMCPY(newPtr, myRawMemory, mySize * sizeof(T));
        operator delete[](myRawMemory);
        myRawMemory = newPtr;
        myCapacity = theCapacity;
    }
}


template<class T>
void JVec<T>::swap(JVec& other) {
    void* tempMem = myRawMemory;
    myRawMemory = other.myRawMemory;
    other.myRawMemory = tempMem;

    unsigned long long tempVar = mySize;
    mySize = other.mySize;
    other.mySize = tempVar;

    tempVar = myCapacity;
    myCapacity = other.myCapacity;
    other.myCapacity = tempVar;
}