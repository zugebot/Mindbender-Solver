#pragma once

void _JVEC_HIDDEN_PRINTF(const char* str, unsigned long long num);
void _JVEC_HIDDEN_MEMCPY(void *dst, const void *src, unsigned long long size);


template<class T>
class [[maybe_unused]] JVec {
    void* myRawMemory{};
    unsigned long long mySize{};
    unsigned long long myCapacity{};

    inline T* ptr_t() { return static_cast<T*>(myRawMemory); }
    inline const T* c_ptr_t() const { return static_cast<T*>(myRawMemory); }

public:
    [[maybe_unused]] explicit JVec();
    [[maybe_unused]] explicit JVec(unsigned long long theCapacity);
    ~JVec();

    JVec(const JVec& other);
    JVec& operator=(const JVec& other);
    JVec(JVec&& other) noexcept;
    JVec& operator=(JVec&& other) noexcept;

    [[nodiscard]] inline unsigned long long size() const { return mySize; }
    [[nodiscard]] inline unsigned long long capacity() const { return myCapacity; }

    [[nodiscard]] inline bool empty() const { return mySize == 0; }
    inline const T* data() const { return c_ptr_t(); }

    inline T* begin() { return ptr_t(); }
    inline T* end() { return ptr_t() + mySize; }

    inline const T* begin() const { return c_ptr_t(); }
    inline const T* end() const { return c_ptr_t() + mySize; }

    inline void clear() { mySize = 0; }
    void resize(unsigned long long theSize);
    void reserve(unsigned long long theCapacity);
    void swap(JVec& other);

    inline T& operator[](unsigned long long index) { return ptr_t()[index]; }
    inline const T& operator[](unsigned long long index) const { return c_ptr_t()[index]; }
};


template<class T>
[[maybe_unused]] JVec<T>::JVec() {
    static constexpr unsigned long long CAPACITY = 0;
    mySize = 0;
    myCapacity = CAPACITY;
    myRawMemory = operator new[](CAPACITY * sizeof(T));
    if (!myRawMemory) {
        _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %llu", CAPACITY);
    }
}


template<class T>
[[maybe_unused]] JVec<T>::JVec(const unsigned long long theCapacity) {
    mySize = 0;
    myCapacity = theCapacity;
    myRawMemory = operator new[](theCapacity * sizeof(T));
    if (!myRawMemory) {
        _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %llu", theCapacity);
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
JVec<T>::JVec(const JVec& other) {
    mySize = other.mySize;
    myCapacity = other.myCapacity;
    if (myCapacity > 0) {
        myRawMemory = operator new[](myCapacity * sizeof(T));
        if (!myRawMemory) {
            _JVEC_HIDDEN_PRINTF("failed to allocate JVec with size %llu", myCapacity);
        }
        _JVEC_HIDDEN_MEMCPY(myRawMemory, other.myRawMemory, mySize * sizeof(T));
    } else {
        myRawMemory = nullptr;
    }
}


template<class T>
JVec<T>& JVec<T>::operator=(const JVec& other) {
    if (this != &other) {
        operator delete[](myRawMemory);
        mySize = other.mySize;
        myCapacity = other.myCapacity;
        if (myCapacity > 0) {
            myRawMemory = operator new[](myCapacity * sizeof(T));
            if (!myRawMemory) {
                _JVEC_HIDDEN_PRINTF("failed to allocate JVec with size %llu", myCapacity);
            }
            _JVEC_HIDDEN_MEMCPY(myRawMemory, other.myRawMemory, mySize * sizeof(T));
        } else {
            myRawMemory = nullptr;
        }
    }
    return *this;
}


template<class T>
JVec<T>::JVec(JVec&& other) noexcept {
    myRawMemory = other.myRawMemory;
    mySize = other.mySize;
    myCapacity = other.myCapacity;
    other.myRawMemory = nullptr;
    other.mySize = 0;
    other.myCapacity = 0;
}


template<class T>
JVec<T>& JVec<T>::operator=(JVec&& other)  noexcept {
    if (this != &other) {
        operator delete[](myRawMemory);
        myRawMemory = other.myRawMemory;
        mySize = other.mySize;
        myCapacity = other.myCapacity;
        other.myRawMemory = nullptr;
        other.mySize = 0;
        other.myCapacity = 0;
    }
    return *this;
}


template<class T>
void JVec<T>::resize(const unsigned long long theSize) {
    if (theSize < myCapacity) {
        mySize = theSize;
    } else {
        void* newPtr = operator new[](theSize * sizeof(T));
        if (!myRawMemory) {
            _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %llu", theSize);
        }
        _JVEC_HIDDEN_MEMCPY(newPtr, myRawMemory, mySize * sizeof(T));
        operator delete[](myRawMemory);
        myRawMemory = newPtr;
        mySize = theSize;
        myCapacity = theSize;
    }
}


template<class T>
void JVec<T>::reserve(const unsigned long long theCapacity) {
    if (theCapacity > myCapacity) {
        void* newPtr = operator new[](theCapacity * sizeof(T));
        if (!myRawMemory) {
            _JVEC_HIDDEN_PRINTF("failed to allocate jVec with size %llu", theCapacity);
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