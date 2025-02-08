import Image from "next/image";
// import Link from "next/link";
import PageName from "../components/Page name bar/pageName";
import ShopBar from "../components/shopbar/shopbar";
import "./grid.css";

export default function GridDefault() {
  return (
    <div>
      <PageName name="Shop Grid Default" />
      <ShopBar />

      <div className="prod-grid flex justify-center items-center">
        <div className="products grid grid-cols-4">
          <div className="prod ">
            <div className="prod-img flex justify-center items-center relative">
              <Image
                src="/grid/prod-1.png"
                alt="Product 1"
                width={201}
                height={201}
              />
              {/* <div className="side-icons absolute">
                <button className="prod-add-cart">
                  <Image
                    src="./icons/cart.svg"
                    alt="cart icon"
                    width={20}
                    height={20}
                  />
                </button>
                <button></button>
                <button></button>
                <button></button>
              </div> */}
            </div>
            <div className="prod-details flex flex-col justify-center items-center">
              <h3>Vel elit euismod</h3>
              <div className="prod-color-btns flex gap-x-2">
                <button className="gold"></button>
                <button className="pink"></button>
                <button className="violet"></button>
              </div>
              <p>
                $26.00 <span>$42.00</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
