/*
#define DIF(func1) static_cast<short>(reinterpret_cast<uint64_t>( \
        ( \
        reinterpret_cast<uintptr_t>(func1) - reinterpret_cast<uintptr_t>(R_0_1) \
        ) \
 ))



class ActionHelper {
public:
    MUND static __forceinline Action getAllAction(size_t index) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myAllActions[index]);
    }

    MUND static __forceinline Action getNormalAction(c_u64 index) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myNormalActions[index]);
    }

    MUND static __forceinline Action getFatAction(c_u64 index1, c_u64 index2) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myFatActions[index1][index2]);
    }


    MU static __forceinline void applyAllAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myAllActions[index]))(board);
    }

    MU static __forceinline void applyNormalAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myNormalActions[index]))(board);
    }


    MU static __forceinline void applyFatAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myFatActions[board.getFatXY()][index]))(board);
    }



private:
    typedef short ptrType;
    static const uintptr_t smallestPtr;

    static const std::array<ptrType, 110> myAllActions;
    static const std::array<ptrType, 60> myNormalActions;
    static const std::array<std::array<ptrType, 48>, 25> myFatActions;


    static std::array<ptrType, 110> init_allActions() {
        std::array<ptrType, 110> arr{};
        for (int i = 0; i < 110; i++) {
            arr[i] = DIF(allActionsList[i]);
        }
        return arr;
    }


    static std::array<ptrType, 60> init_normalActions() {
        std::array<ptrType, 60> arr{};
        for (int i = 0; i < 60; i++) {
            arr[i] = DIF(actions[i]);
        }
        return arr;
    }

    static std::array<std::array<ptrType, 48>, 25> init_fatActions() {
        std::array<std::array<ptrType, 48>, 25> arr{};
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 48; j++) {
                arr[i][j] = DIF(fatActions[i][j]);
            }
        }
        return arr;
    }
};
*/


/*
const uintptr_t ActionHelper::smallestPtr = reinterpret_cast<uintptr_t>(R_0_1);
const std::array<ActionHelper::ptrType, 110> ActionHelper::myAllActions = ActionHelper::init_allActions();
const std::array<ActionHelper::ptrType, 60> ActionHelper::myNormalActions = ActionHelper::init_normalActions();
const std::array<std::array<ActionHelper::ptrType, 48>, 25> ActionHelper::myFatActions = ActionHelper::init_fatActions();
*/