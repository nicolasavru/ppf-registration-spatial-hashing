#ifndef CYCLE_ITERATOR_H
#define CYCLE_ITERATOR_H

#include <boost/iterator/iterator_adaptor.hpp>

// http://stackoverflow.com/questions/1782019/easiest-way-to-make-a-cyclic-iterator-circulator
template<class BaseIterator>
class cycle_iterator
    : public boost::iterator_adaptor<
    cycle_iterator<BaseIterator>,  // Derived
    BaseIterator,                  // Base
    boost::use_default,            // Value
    boost::forward_traversal_tag   // CategoryOrTraversal
    >{
  private:
    BaseIterator m_itBegin;
    BaseIterator m_itEnd;

  public:
    cycle_iterator(BaseIterator itBegin, BaseIterator itEnd)
        : cycle_iterator::iterator_adaptor_(itBegin),
        m_itBegin(itBegin),
        m_itEnd(itEnd){
    }

    void increment(){
        if(this->base_reference() == m_itEnd){
            this->base_reference() = m_itBegin;
        }
        else{
            ++this->base_reference(); // increments the iterator we actually point at
        }
    }

    // implement decrement() and advance() if necessary
};


#endif /* CYCLE_ITERATOR_H */
