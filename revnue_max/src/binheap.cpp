#ifndef BINHEAP_CPP
#define BINHEAP_CPP

#include <vector>
using namespace std;

class MinHeap
{
public:
  vector<uint32_t> _vector;
  vector<uint32_t> _vindex;
  vector<int> _vloc;
  
  void BubbleDown(int index);
  void BubbleUp(int index);
  void Heapify();

  MinHeap(uint32_t* array, size_t length);
  MinHeap(const vector<uint32_t>& vector, size_t length);
  MinHeap();
  MinHeap( size_t nNodes );

  void Insert(uint32_t, uint32_t);
  uint32_t GetMin();
  uint32_t extractNode();
  void DeleteMin();
  void DecreaseValue( int loc, int newValue );
  size_t size();

  bool present( uint32_t node ) {
    if(_vloc[node] != -1)
      return true;

    return false;
  }

  void IncreaseValue( int loc, int newValue ) {
    _vector[ _vloc[ loc ] ] = newValue;
    BubbleDown( _vloc[ loc ] );
  }
};

//////

MinHeap::MinHeap(uint32_t* array, size_t length) : _vector(length), _vindex(length), _vloc( length )
{
  for(size_t i = 0; i < length; ++i)
    {
      _vloc[i] = i;
      _vindex[i] = i;
      _vector[i] = array[i];
    }

  Heapify();
}

MinHeap::MinHeap(const vector<uint32_t>& vector, size_t length ) :
  _vector(length), _vindex(length), _vloc( length )  {

  for(size_t i = 0; i < length; ++i) {
    _vloc[i] = i;
    _vindex[i] = i;
    _vector[i] = vector[i];
  }
  
  Heapify();
}

MinHeap::MinHeap()
{
}

/*
 * Create empty heap, but
 * with _vloc tracking node positions
 */
MinHeap::MinHeap( size_t nNodes ) {
  _vloc.assign( nNodes, -1 );
}

void MinHeap::DecreaseValue( int loc,
			     int newValue ) {
  //  if (_vloc[ loc ] >= 0) {
    _vector[ _vloc[ loc ] ] = newValue;
    BubbleUp( _vloc[ loc ] );
    //  }

}
			    

size_t MinHeap::size() {
  return _vector.size();
}

void MinHeap::Heapify()
{
  int length = _vector.size();
  for(int i=length-1; i>=0; --i)
    {
      BubbleDown(i);
    }
}

void MinHeap::BubbleDown(int index)
{
  int length = _vector.size();
  int leftChildIndex = 2*index + 1;
  int rightChildIndex = 2*index + 2;

  if(leftChildIndex >= length)
    return; //index is a leaf

  int minIndex = index;

  if(_vector[index] > _vector[leftChildIndex])
    {
      minIndex = leftChildIndex;
    }

  if((rightChildIndex < length) && (_vector[minIndex] > _vector[rightChildIndex]))
    {
      minIndex = rightChildIndex;
    }

  if(minIndex != index)
    {
      //need to swap
      swap( _vector[index], _vector[minIndex] );
      swap( _vloc[ _vindex[ index ] ], _vloc[ _vindex[ minIndex ] ] );
      swap( _vindex[index], _vindex[minIndex] );
      BubbleDown(minIndex);
    }
}

void MinHeap::BubbleUp(int index)
{
  if(index == 0)
    return;

  int parentIndex = (index-1)/2;

  if(_vector[parentIndex] > _vector[index])
    {
      swap( _vector[ parentIndex ], _vector[ index ] );
      swap( _vindex[ parentIndex ], _vindex[ index ] );
      swap( _vloc[ _vindex[ parentIndex ] ], _vloc[ _vindex[ index ] ] );
      BubbleUp(parentIndex);
    }
}

/*
 * 
 */
void MinHeap::Insert(uint32_t node, uint32_t newValue)
{
  if (_vloc[ node ] != -1) {
    //node's distance is already in queue
    DecreaseValue( node, newValue );
  } else {
    //This node is being rediscovered after it was previously extracted
    //Need to add it back in
    _vector.push_back( newValue );
    _vindex.push_back( node );
    _vloc[ node ] = _vector.size() - 1;
    
    BubbleUp(_vector.size() - 1);
  }
}

uint32_t MinHeap::GetMin()
{
  return _vector.front();
}

uint32_t MinHeap::extractNode() {
  uint32_t r = _vindex.front(); //node id of min. distance
  DeleteMin();
  return r;
}


void MinHeap::DeleteMin()
{
  int length = _vector.size();

  if(length == 0)
    {
      return;
    }

  _vector[0] = _vector[length-1];
  
  _vloc[ _vindex[ 0 ] ] = -1;

  _vindex[ 0 ] = _vindex[ length - 1];
  _vindex.pop_back();
  
  _vloc[ _vindex[ 0 ] ] = 0;
  
  _vector.pop_back();

  BubbleDown(0);
}


#endif
