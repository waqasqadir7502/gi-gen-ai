import Image from "next/image";
import "./faq.css";
import PageName from "../components/Page name bar/pageName";

export default function Faq() {
  return (
    <div>
      <PageName name="FAQ" />
      <div className="faq-sec flex justify-center items-center">
        <div className="faq-inner grid grid-cols-2">
          <div className="faq-col-1 flex flex-col ">
            <h3>General Information</h3>

            <div className="faq-questions ">
              <p className="question">
                Eu dictumst cum at sed euismood condimentum?
              </p>
              <p className="answer">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                Tincidunt sed <br />tristique mollis vitae, consequat gravida
                sagittis.
              </p>
            </div>

            <div className="faq-questions">
              <p className="question">Magna bibendum est fermentum eros.</p>
              <p className="answer">Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                Tincidunt sed <br /> tristique mollis vitae, consequat gravida
                sagittis.</p>
            </div>

            <div className="faq-questions">
              <p className="question">Odio muskana hak eris conseekin sceleton?</p>
              <p className="answer">Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                Tincidunt sed <br />tristique mollis vitae, consequat gravida
                sagittis.</p>
            </div>

            <div className="faq-questions">
              <p className="question">Elit id blandit sabara boi velit gua mara?</p>
              <p className="answer">Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                Tincidunt sed <br />tristique mollis vitae, consequat gravida
                sagittis.</p>
            </div>
          </div>

          <div className="faq-col-2 flex flex-col justify-between ">
            <h3>Ask A Question</h3>
            <form className="faq-form flex flex-col items-start">
              <input type="text" placeholder="Your Name*"/>
              <input type="text" placeholder="Subject"/>
              <textarea placeholder="Type Your Message!"></textarea>
              <button>Send Mail</button>
            </form>
          </div>

        </div>
      </div>
    
      <div className="brand-banner flex justify-center items-center mt-5 mb-10">
        <Image
          src="/homepage/partnered-firms.png"
          alt=""
          width={1246}
          height={128}
        />
      </div>

    </div>
  );
}
