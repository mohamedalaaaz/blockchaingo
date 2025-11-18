package main

import (
	"fmt"
	"log"
	"time"
)

type Block struct {
	nonce        int
	pervioushash string
	timestamp    int64
	transction   []string
}

func Newblock(nonce int, pervioushash string) *Block {
	b := new(Block)
	b.timestamp = time.Now().UnixNano()
	b.nonce = nonce
	b.pervioushash = pervioushash
	return b
}

func (b *Block) print() {
	fmt.Println("Block hash:", b.pervioushash)
	fmt.Println("Timestamp:", b.timestamp)
	fmt.Println("Nonce:", b.nonce)

}

type Blockchain struct {
	transaction []string
	chain       []*Block
}

func newBlockchain() *Blockchain {
	bc := new(Blockchain)
	bc.CreateBlock(0, "init hash")
	return bc
}
func (bc *Blockchain) CreateBlock(nonce int, pervioushash string) *Block {
	b := Newblock(nonce, pervioushash)
	bc.chain = append(bc.chain, b)
	return b
}

func (bc *Blockchain) Print() {
	for i, block := range bc.chain {
		fmt.Println("chain  \n", i)
		block.print()
	}
}

func init() {
	log.SetPrefix("blockcahin: ")

}
func main() {
	blockchain := newBlockchain()
	blockchain.Print()

}
