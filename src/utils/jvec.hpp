#pragma once

#include <cstdlib>
#include <initializer_list>
#include <new>
#include <type_traits>

void _JVEC_HIDDEN_PRINTF(const char* str, unsigned long long num);
void _JVEC_HIDDEN_MEMCPY(void* dst, const void* src, unsigned long long size);

template<class T>
class [[maybe_unused]] JVec {
    static_assert(std::is_trivially_copyable_v<T>,
                  "JVec requires trivially copyable T");
    static_assert(std::is_trivially_destructible_v<T>,
                  "JVec requires trivially destructible T");

    void* myRawMemory{};
    unsigned long long mySize{};
    unsigned long long myCapacity{};

    inline T* ptr_t() { return static_cast<T*>(myRawMemory); }
    inline const T* c_ptr_t() const { return static_cast<const T*>(myRawMemory); }

    static void* alloc_bytes(unsigned long long count) {
        if (count == 0) {
            return nullptr;
        }

        void* p = std::malloc(count * sizeof(T));
        if (!p) {
            _JVEC_HIDDEN_PRINTF("failed to allocate JVec with size %llu", count);
            throw std::bad_alloc();
        }
        return p;
    }

    static void* realloc_bytes(void* oldPtr, unsigned long long count) {
        if (count == 0) {
            std::free(oldPtr);
            return nullptr;
        }

        void* p = std::realloc(oldPtr, count * sizeof(T));
        if (!p) {
            _JVEC_HIDDEN_PRINTF("failed to reallocate JVec with size %llu", count);
            throw std::bad_alloc();
        }
        return p;
    }

public:
    [[maybe_unused]] JVec();
    [[maybe_unused]] explicit JVec(unsigned long long theCapacity);
    [[maybe_unused]] JVec(std::initializer_list<T> theInitList);
    ~JVec();

    JVec(const JVec& other);
    JVec& operator=(const JVec& other);
    JVec(JVec&& other) noexcept;
    JVec& operator=(JVec&& other) noexcept;

    [[nodiscard]] inline unsigned long long size() const { return mySize; }
    [[nodiscard]] inline unsigned long long capacity() const { return myCapacity; }

    [[nodiscard]] inline bool empty() const { return mySize == 0; }
    inline const T* data() const { return c_ptr_t(); }

    inline T* data() { return ptr_t(); }

    inline T* begin() { return ptr_t(); }
    inline T* end() { return ptr_t() + mySize; }

    inline const T* begin() const { return c_ptr_t(); }
    inline const T* end() const { return c_ptr_t() + mySize; }

    inline void clear() { mySize = 0; }
    void resize(unsigned long long theSize);
    void reserve(unsigned long long theCapacity);
    void shrink_to_fit();
    void swap(JVec& other);

    inline T& operator[](unsigned long long index) { return ptr_t()[index]; }
    inline const T& operator[](unsigned long long index) const { return c_ptr_t()[index]; }
};

template<class T>
[[maybe_unused]] JVec<T>::JVec() {
    myRawMemory = nullptr;
    mySize = 0;
    myCapacity = 0;
}

template<class T>
[[maybe_unused]] JVec<T>::JVec(const unsigned long long theCapacity) {
    mySize = 0;
    myCapacity = theCapacity;
    myRawMemory = alloc_bytes(myCapacity);
}

template<class T>
[[maybe_unused]] JVec<T>::JVec(std::initializer_list<T> theInitList) {
    mySize = theInitList.size();
    myCapacity = theInitList.size();
    myRawMemory = alloc_bytes(myCapacity);

    if (mySize != 0) {
        unsigned long long i = 0;
        for (const T& value : theInitList) {
            ptr_t()[i++] = value;
        }
    }
}

template<class T>
[[maybe_unused]] JVec<T>::~JVec() {
    std::free(myRawMemory);
    myRawMemory = nullptr;
    mySize = 0;
    myCapacity = 0;
}

template<class T>
JVec<T>::JVec(const JVec& other) {
    mySize = other.mySize;
    myCapacity = other.mySize;
    myRawMemory = alloc_bytes(myCapacity);

    if (mySize != 0) {
        _JVEC_HIDDEN_MEMCPY(myRawMemory, other.myRawMemory, mySize * sizeof(T));
    }
}

template<class T>
JVec<T>& JVec<T>::operator=(const JVec& other) {
    if (this != &other) {
        void* newMemory = alloc_bytes(other.mySize);

        if (other.mySize != 0) {
            _JVEC_HIDDEN_MEMCPY(newMemory, other.myRawMemory, other.mySize * sizeof(T));
        }

        std::free(myRawMemory);
        myRawMemory = newMemory;
        mySize = other.mySize;
        myCapacity = other.mySize;
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
JVec<T>& JVec<T>::operator=(JVec&& other) noexcept {
    if (this != &other) {
        std::free(myRawMemory);

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
    if (theSize <= myCapacity) {
        mySize = theSize;
        return;
    }

    myRawMemory = realloc_bytes(myRawMemory, theSize);
    mySize = theSize;
    myCapacity = theSize;
}

template<class T>
void JVec<T>::reserve(const unsigned long long theCapacity) {
    if (theCapacity <= myCapacity) {
        return;
    }

    myRawMemory = realloc_bytes(myRawMemory, theCapacity);
    myCapacity = theCapacity;
}

template<class T>
void JVec<T>::shrink_to_fit() {
    if (mySize == myCapacity) {
        return;
    }

    myRawMemory = realloc_bytes(myRawMemory, mySize);
    myCapacity = mySize;
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