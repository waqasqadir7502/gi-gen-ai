import Image from "next/image";
import "./order-complete.css";
import PageName from "../components/Page name bar/pageName";

export default function OrderComplete() {
  return (
    <div>
      <PageName name="Order Complete" />
      <div className="order-comp-sec flex justify-center items-center">
        <div className="order-comp-inner">
          <div className="complete-display flex flex-col items-center">
            <div className="order-comp-img">
              <div className="comp-img-inner flex justify-center">
                <Image
                  src="/ordercomplete/order-complete.png"
                  alt="Order Complete pic"
                  width={88}
                  height={81}
                />
              </div>
            </div>
            <h2 className="order-comp-title">Your Order Is Completed! </h2>
            <div className="inner-2 flex flex-col items-center">
              <p className="order-comp-para">
                Thank you for your order! Your order is being processed and will
                be completed within 3-6 <br /> hours. You will receive an email
                confirmation when your order is completed.
              </p>
              <button className="continue-btn">Continue Shopping</button>
            </div>
          </div>
        </div>
      </div>

      <div className="brand-banner flex justify-center items-center">
        <Image
          src="/homepage/partnered-firms.png"
          alt=""
          width={1176}
          height={121}
        />
      </div>
    </div>
  );
}
